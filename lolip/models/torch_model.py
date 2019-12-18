import os
from functools import partial

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import MultiStepLR

from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent

import numpy as np
from sklearn.base import BaseEstimator
from .torch_utils import get_optimizer, get_loss
from .torch_utils.archs import *
from ..attacks.torch.projected_gradient_descent import projected_gradient_descent

DEBUG = int(os.getenv("DEBUG", 0))


class TorchModel(BaseEstimator):
    def __init__(self, lbl_enc, n_features, n_classes, loss_name='ce',
                learning_rate=1e-4, momentum=0.0, batch_size=256, epochs=20,
                optimizer='adam', architecture='arch_001', random_state=None,
                callbacks=None, train_type:str=None, eps:float=0.1, norm=np.inf):
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.epochs = epochs
        self.lbl_enc = lbl_enc
        self.loss_name = loss_name

        model = globals()[self.architecture](self.n_classes)
        if torch.cuda.is_available():
            model = model.cuda()

        self.optimizer = get_optimizer(model, optimizer, learning_rate, momentum)
        self.model = model

        self.callbacks=callbacks
        self.random_state = random_state
        self.train_type = train_type

        ### Attack ####
        self.eps = eps
        self.norm = norm
        ###############

    def _get_dataset(self, X, y=None):
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
        loss_fn = get_loss(self.loss_name)
        X = self._preprocess_x(X)

        dataset = self._get_dataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=True, num_workers=2)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        scheduler = MultiStepLR(self.optimizer, milestones=[75,90,100], gamma=0.1)

        self.model.train()
        for epoch in range(1, self.epochs+1):
            train_loss = 0.
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                if 'adv' in self.loss_name:
                    x = projected_gradient_descent(self.model, x, y=y,
                        eps_iter=0.01, eps=self.eps, norm=self.norm, nb_iter=40)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = loss_fn(output, y)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                scheduler.step()
            if (epoch - 1) % log_interval == 0:
                print('epoch: {}/{}, train loss: {:.3f}'.format(epoch, self.epochs, train_loss))

    def predict(self, X):
        self.model.eval()
        dataset = data_utils.TensorDataset(torch.from_numpy(X).float())
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=2)

        x = torch.from_numpy(X).float()
        _, y_pred = self.model(x).max(1)
        return y_pred.numpy()

    def predict_proba(self, X):
        self.model.eval()
        x = torch.from_numpy(X).float()
        return self.model(x).detach().numpy()