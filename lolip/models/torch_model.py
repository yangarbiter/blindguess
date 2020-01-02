import gc
import os
from functools import partial

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
from sklearn.base import BaseEstimator
from .torch_utils import get_optimizer, get_loss, get_scheduler
from .torch_utils.archs import *
from ..attacks.torch.projected_gradient_descent import projected_gradient_descent
from .torch_utils.trades import trades_loss
from .torch_utils.llr import locally_linearity_regularization
from .torch_utils.cure import cure_loss
from .torch_utils.gradient_regularization import gradient_regularization

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
        self.start_epoch = 1

        ### Attack ####
        self.eps = eps
        self.norm = norm
        ###############

    def _get_dataset(self, X, y=None):
        X = self._preprocess_x(X)
        if y is None:
            return torch.utils.data.TensorDataset(torch.from_numpy(X).float())
        if 'mse' in self.loss_name:
            Y = self.lbl_enc.transform(y.reshape(-1, 1))
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        else:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        return dataset

    def _preprocess_x(self, X):
        return X.transpose(0, 3, 1, 2)

    def fit(self, X, y, sample_weight=None):
        verbose = 0 if not DEBUG else 1
        log_interval = 1

        history = []
        loss_fn = get_loss(self.loss_name)
        scheduler = get_scheduler(self.optimizer, n_epochs=self.epochs)

        dataset = self._get_dataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=True, num_workers=1)

        test_loader = None
        if self.tst_ds is not None:
            tstX, tsty = self.tst_ds
            dataset = self._get_dataset(tstX, tsty)
            test_loader = torch.utils.data.DataLoader(dataset,
                batch_size=16, shuffle=False, num_workers=1)

        for epoch in range(self.start_epoch, self.epochs+1):
            train_loss = 0.
            train_acc = 0.
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)

                if 'trades' in self.loss_name:
                    if 'trades10' in self.loss_name:
                        beta = 10.0
                    elif 'trades6' in self.loss_name:
                        beta = 6.0
                    else:
                        beta = 1.0
                    
                    if 'K20' in self.loss_name:
                        steps = 20
                    else:
                        steps = 10
                    
                    version = None
                    if 'ptrades' in self.loss_name:
                        version = "plus"

                    outputs, loss = trades_loss(
                        self.model, loss_fn, x, y, norm=self.norm, optimizer=self.optimizer,
                        step_size=self.eps*2/steps, epsilon=self.eps, perturb_steps=steps, beta=beta,
                        version=version, device=self.device
                    )
                elif 'llr' in self.loss_name:
                    if 'llr65' in self.loss_name:
                        lambd, mu = 6.0, 5.0
                    else:
                        lambd, mu = 4.0, 3.0
                    outputs, loss = locally_linearity_regularization(
                        self.model, loss_fn, x, y, norm=self.norm, optimizer=self.optimizer,
                        step_size=self.eps/5, epsilon=self.eps, perturb_steps=10,
                        lambd=lambd, mu=mu
                    )
                elif 'cure' in self.loss_name:
                    if 'cure68' in self.loss_name:
                        h, lambda_ = 6.0, 8.0
                    elif 'cure14' in self.loss_name:
                        h, lambda_ = 1.25, 4.0
                    else:
                        h, lambda_ = 3.0, 4.0

                    self.optimizer.zero_grad()
                    outputs, loss = cure_loss(self.model, loss_fn, x, y, h=h, lambda_=lambda_)
                elif 'gr' in self.loss_name:
                    if 'gr4' in self.loss_name:
                        lambd = 4.0
                    elif 'gr1e3' in self.loss_name:
                        lambd = 1e3
                    elif 'gr1e2' in self.loss_name:
                        lambd = 1e2
                    else:
                        lambd = 1.0
                    outputs, loss = gradient_regularization(
                        self.model, loss_fn, self.optimizer, x, y, lambd=lambd)
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
            self.start_epoch = epoch

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
                    with torch.no_grad():
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
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
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
        #torch.save(self.model.state_dict(), path)
        torch.save({
            'epoch': self.start_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path % self.start_epoch)

    def load(self, path):
        self.model = globals()[self.architecture](self.n_classes)
        loaded = torch.load(path)
        if 'epoch' in loaded:
            self.start_epoch = loaded['epoch']
            self.model.load_state_dict(loaded['model_state_dict'])
            self.optimizer.load_state_dict(loaded['optimizer_state_dict'])
        else:
            self.model.load_state_dict(torch.load(path))
        self.model.eval()
