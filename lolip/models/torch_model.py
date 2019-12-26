import gc
import os
from functools import partial

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import numpy as np
from sklearn.base import BaseEstimator
from .torch_utils import get_optimizer, get_loss
from .torch_utils.archs import *
from ..attacks.torch.projected_gradient_descent import projected_gradient_descent
from .torch_utils.trades import trades_loss
from .torch_utils.llr import locally_linearity_regularization
from .torch_utils.cure import cure_loss

DEBUG = int(os.getenv("DEBUG", 0))


class TorchModel(BaseEstimator):
    def __init__(self, lbl_enc, n_features, n_classes, loss_name='ce',
                learning_rate=1e-4, momentum=0.0, batch_size=256, epochs=20,
                optimizer='sgd', architecture='arch_001', random_state=None,
                callbacks=None, train_type=None, eps:float=0.1, norm=np.inf):
        print(f'lr: {learning_rate}, opt: {optimizer}, loss: {loss_name}, '
              f'arch: {architecture}')
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.epochs = epochs
        self.lbl_enc = lbl_enc
        self.loss_name = loss_name

        model = globals()[self.architecture](n_classes=self.n_classes)
        if torch.cuda.is_available():
            model = model.cuda()

        self.optimizer = get_optimizer(model, optimizer, learning_rate, momentum)
        self.model = model

        #self.callbacks=callbacks
        self.random_state = random_state
        self.train_type = train_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tst_ds = None

        ### Attack ####
        self.eps = eps
        self.norm = norm
        ###############

    def _get_dataset(self, X, y=None):
        X = self._preprocess_x(X)
        if y is None:
            return data_utils.TensorDataset(torch.from_numpy(X).float())
        if 'mse' in self.loss_name:
            Y = self.lbl_enc.transform(y.reshape(-1, 1))
            dataset = data_utils.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        else:
            dataset = data_utils.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        return dataset

    def _preprocess_x(self, X):
        return X.transpose(0, 3, 1, 2)

    def fit(self, X, y, sample_weight=None):
        verbose = 0 if not DEBUG else 1
        log_interval = 1

        history = []
        loss_fn = get_loss(self.loss_name)
        scheduler = MultiStepLR(self.optimizer, milestones=[60, 100, 140, 180], gamma=0.1)

        dataset = self._get_dataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=True, num_workers=1)

        test_loader = None
        if self.tst_ds is not None:
            tstX, tsty = self.tst_ds
            dataset = self._get_dataset(tstX, tsty)
            test_loader = torch.utils.data.DataLoader(dataset,
                batch_size=16, shuffle=False, num_workers=1)

        for epoch in range(1, self.epochs+1):
            train_loss = 0.
            train_acc = 0.
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)

                if 'trades6' in self.loss_name:
                    outputs, loss = trades_loss(
                        self.model, loss_fn, x, y, norm=self.norm, optimizer=self.optimizer,
                        step_size=self.eps/5, epsilon=self.eps, perturb_steps=10, beta=6.0
                    )
                elif 'trades' in self.loss_name:
                    outputs, loss = trades_loss(
                        self.model, loss_fn, x, y, norm=self.norm, optimizer=self.optimizer,
                        step_size=self.eps/5, epsilon=self.eps, perturb_steps=10, beta=1.0
                    )
                elif 'llr' in self.loss_name:
                    outputs, loss = locally_linearity_regularization(
                        self.model, loss_fn, x, y, norm=self.norm, optimizer=self.optimizer,
                        step_size=self.eps/5, epsilon=self.eps, perturb_steps=10,
                        lambd=4.0, mu=3.0
                    )
                elif 'cure' in self.loss_name:
                    outputs, loss = cure_loss(
                        self.model, loss_fn, x, y, h=3, lambda_=4, device=self.device)
                else:
                    if 'adv' in self.loss_name:
                        x = projected_gradient_descent(self.model, x, y=y,
                                clip_min=0, clip_max=1, eps_iter=self.eps/5,
                                eps=self.eps, norm=self.norm, nb_iter=10)
                    self.optimizer.zero_grad()
                    outputs = self.model(x)
                    #outputs = F.softmax(self.model(x), dim=1)
                    loss = loss_fn(outputs, y)

                loss.backward()
                self.optimizer.step()

                if (epoch - 1) % log_interval == 0:
                    self.model.eval()
                    train_loss += loss.item()
                    train_acc += (outputs.argmax(dim=1)==y).sum().float().item()

            scheduler.step()

            if (epoch - 1) % log_interval == 0:
                self.model.eval()
                history.append({
                    'epoch': epoch,
                    'trn_loss': train_loss / len(train_loader.dataset),
                    'trn_acc': train_acc / len(train_loader.dataset),
                })
                print('epoch: {}/{}, train loss: {:.3f}, train acc: {:.3f}'.format(
                    epoch, self.epochs, history[-1]['trn_loss'], history[-1]['trn_acc']))

                if self.tst_ds is not None:
                    tst_loss, tst_acc = 0., 0.
                    for tx, ty in test_loader:
                        tx, ty = tx.to(self.device), ty.to(self.device)
                        outputs = self.model(tx)
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
    
    def _prep_pred(self, X):
        X = self._preprocess_x(X)
        self.model.eval()
        dataset = data_utils.TensorDataset(torch.from_numpy(X).float())
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=2)
        return loader

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
            output = F.softmax(self.model(x.to(self.device)).detach())
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
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model = globals()[self.architecture](self.n_classes)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
