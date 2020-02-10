import gc
import os
from functools import partial
import inspect

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import VisionDataset

import numpy as np
from sklearn.base import BaseEstimator
from .torch_utils.losses import get_outputs_loss
from .torch_utils import get_optimizer, get_loss, get_scheduler, CustomTensorDataset
from .torch_utils import archs, data_augs

DEBUG = int(os.getenv("DEBUG", 0))


class TorchModelV2(BaseEstimator):
    def __init__(self, lbl_enc, n_features, n_classes, loss_name='ce',
                n_channels=None, learning_rate=1e-4, momentum=0.0, weight_decay=0.0,
                batch_size=256, epochs=20, optimizer='sgd', architecture='arch_001',
                random_state=None, callbacks=None, train_type=None, eps:float=0.1, norm=np.inf,
                multigpu=False, dataaug=None, device=None, num_workers=4, trn_log_callbacks=None):
        print(f'lr: {learning_rate}, opt: {optimizer}, loss: {loss_name}, '
              f'arch: {architecture}, dataaug: {dataaug}, batch_size: {batch_size}, '
              f'momentum: {momentum}, weight_decay: {weight_decay}, eps: {eps}, '
              f'epochs: {epochs}')
        self.num_workers = num_workers
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.epochs = epochs
        self.lbl_enc = lbl_enc
        self.loss_name = loss_name
        self.dataaug = dataaug

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        arch_fn = getattr(archs, self.architecture)
        if 'n_features' in inspect.getfullargspec(arch_fn)[0]:
            model = arch_fn(n_features=n_features, n_classes=self.n_classes, n_channels=n_channels)
        else:
            model = arch_fn(n_classes=self.n_classes, n_channels=n_channels)
        if self.device == 'cuda':
            model = model.cuda()

        self.multigpu = multigpu
        if self.multigpu:
            model = torch.nn.DataParallel(model, device_ids=[0, 1])

        #if 'rbfw' in self.loss_name:
        #    self.gamma_var = model.gamma_var
        #    self.optimizer = get_optimizer(model, optimizer, learning_rate, momentum, additional_vars=[self.gamma_var])
        #else:
        self.optimizer = get_optimizer(model, optimizer, learning_rate, momentum, weight_decay)
        self.model = model

        #self.callbacks=callbacks
        self.random_state = random_state
        #self.train_type = train_type

        self.tst_ds = None
        self.start_epoch = 1

        ### Attack ####
        self.eps = eps
        self.norm = norm
        ###############

    def _get_dataset(self, X, y=None, sample_weights=None):
        X = self._preprocess_x(X)
        if sample_weights is None:
            sample_weights = np.ones(len(X))

        if self.dataaug is None:
            transform = None
        else:
            if y is None:
                transform = getattr(data_augs, self.dataaug)()[1]
            else:
                transform = getattr(data_augs, self.dataaug)()[0]

        if y is None:
            return CustomTensorDataset((torch.from_numpy(X).float(), ), transform=transform)
        if 'mse' in self.loss_name:
            Y = self.lbl_enc.transform(y.reshape(-1, 1))
            dataset = CustomTensorDataset(
                (torch.from_numpy(X).float(), torch.from_numpy(Y).float(), torch.from_numpy(sample_weights).float()), transform=transform)
        else:
            dataset = CustomTensorDataset(
                (torch.from_numpy(X).float(), torch.from_numpy(y).long(), torch.from_numpy(sample_weights).float()), transform=transform)
        return dataset

    def _preprocess_x(self, X):
        if len(X.shape) ==4:
            return X.transpose(0, 3, 1, 2)
        else:
            return X

    def fit_dataset(self, dataset, verbose=None):
        if verbose is None:
            verbose = 0 if not DEBUG else 1
        log_interval = 1

        history = []
        loss_fn = get_loss(self.loss_name, reduction="none")
        scheduler = get_scheduler(self.optimizer, n_epochs=self.epochs, loss_name=self.loss_name)

        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        test_loader = None
        if self.tst_ds is not None:
            if isinstance(self.tst_ds, VisionDataset):
                ts_dataset = self.tst_ds
            else:
                tstX, tsty = self.tst_ds
                ts_dataset = self._get_dataset(tstX, tsty)

            test_loader = torch.utils.data.DataLoader(ts_dataset,
                batch_size=32, shuffle=False, num_workers=self.num_workers)

        for epoch in range(self.start_epoch, self.epochs+1):
            train_loss = 0.
            train_acc = 0.
            for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
                self.model.train()
                x, y, w = (d.to(self.device) for d in data)

                params = {
                    'norm': self.norm,
                    'device': self.device,
                    'eps': self.eps,
                    'clip_img': True if len(x.shape) > 2 else False,
                    'loss_name': self.loss_name,
                    'reduction': 'mean',
                }
                outputs, loss = get_outputs_loss(
                    self.model, self.optimizer, loss_fn, x, y, **params
                )

                loss = (w * loss).mean()

                loss.backward()
                self.optimizer.step()

                if (epoch - 1) % log_interval == 0:
                    self.model.eval()
                    train_loss += loss.item()
                    train_acc += (outputs.argmax(dim=1)==y).sum().float().item()

            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            scheduler.step()
            self.start_epoch = epoch

            if (epoch - 1) % log_interval == 0:
                if 'rbfw' in self.loss_name:
                    print(f"current gamma: {self.gamma_var}")
                print(f"current LR: {current_lr}")
                self.model.eval()
                history.append({
                    'epoch': epoch,
                    'lr': current_lr,
                    'trn_loss': train_loss / len(train_loader.dataset),
                    'trn_acc': train_acc / len(train_loader.dataset),
                })
                print('epoch: {}/{}, train loss: {:.3f}, train acc: {:.3f}'.format(
                    epoch, self.epochs, history[-1]['trn_loss'], history[-1]['trn_acc']))

                if self.tst_ds is not None:
                    tst_loss, tst_acc = 0., 0.
                    with torch.no_grad():
                        for tx, ty, _ in test_loader:
                            tx, ty = tx.to(self.device), ty.to(self.device)
                            outputs = self.model(tx)
                            if loss_fn.reduction == 'none':
                                loss = torch.sum(loss_fn(outputs, ty))
                            else:
                                loss = loss_fn(outputs, ty)
                            tst_loss += loss.item()
                            tst_acc += (outputs.argmax(dim=1)==ty).sum().float().item()
                    history[-1]['tst_loss'] = tst_loss / len(test_loader.dataset)
                    history[-1]['tst_acc'] = tst_acc / len(test_loader.dataset)
                    print('             test loss: {:.3f}, test acc: {:.3f}'.format(
                          history[-1]['tst_loss'], history[-1]['tst_acc']))

        if test_loader is not None:
            del test_loader
        del train_loader
        gc.collect()

        return history

    def fit(self, X, y, sample_weights=None, verbose=None):
        dataset = self._get_dataset(X, y, sample_weights)
        return self.fit_dataset(dataset, verbose=verbose)

    def _prep_pred(self, X):
        if isinstance(X, VisionDataset):
            dataset = X
        else:
            if self.dataaug is None:
                transform = None
            else:
                transform = getattr(data_augs, self.dataaug)()[1]
            X = self._preprocess_x(X)
            self.model.eval()
            dataset = CustomTensorDataset((torch.from_numpy(X).float(), ), transform=transform)
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def predict_ds(self, ds):
        loader = torch.utils.data.DataLoader(ds,
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        ret = []
        for x in loader:
            x = x[0]
            ret.append(self.model(x.to(self.device)).argmax(1).cpu().numpy())
        del loader
        return np.concatenate(ret)

    def predict(self, X):
        loader = self._prep_pred(X)
        ret = []
        for [x] in loader:
            ret.append(self.model(x.to(self.device)).argmax(1).cpu().numpy())
        del loader
        return np.concatenate(ret)

    def predict_proba(self, X):
        loader = self._prep_pred(X)
        ret = []
        for [x] in loader:
            output = F.softmax(self.model(x.to(self.device)).detach(), dim=1)
            ret.append(output.cpu().numpy())
        del loader
        return np.concatenate(ret, axis=0)

    def predict_real(self, X):
        loader = self._prep_pred(X)
        ret = []
        for [x] in loader:
            ret.append(self.model(x.to(self.device)).detach().cpu().numpy())
        del loader
        return np.concatenate(ret, axis=0)

    def save(self, path):
        if self.multigpu:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        torch.save({
            'epoch': self.start_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path % self.start_epoch)

    def load(self, path):
        loaded = torch.load(path)
        if 'epoch' in loaded:
            self.start_epoch = loaded['epoch']
            self.model.load_state_dict(loaded['model_state_dict'])
            self.optimizer.load_state_dict(loaded['optimizer_state_dict'])
        else:
            self.model.load_state_dict(loaded)
        self.model.eval()
