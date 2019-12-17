import itertools
import os
from functools import partial

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data_utils

from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent

import numpy as np
from sklearn.base import BaseEstimator
from .torch_utils import get_optimizer, get_loss
from .torch_utils.archs import *

DEBUG = int(os.getenv("DEBUG", 0))


class TorchModel(BaseEstimator):
    def __init__(self, lbl_enc, n_features, n_classes, loss_name='ce', learning_rate=1e-4,
                momentum=0.0, batch_size=256, epochs=20, optimizer='adam', l2_weight=0,
                architecture='arch_001', random_state=None, attacker=None,
                callbacks=None, train_type:str=None, eps:float=0.1, ord=np.inf,
                attack_method:str = None):
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.epochs = epochs
        self.lbl_enc = lbl_enc
        self.loss_name = loss_name

        model = globals()[self.architecture](self.n_features[0], self.n_classes)
        if torch.cuda.is_available():
            model = model.cuda()

        self.optimizer = get_optimizer(model, optimizer, learning_rate, momentum)

        self.l2_weight = l2_weight
        self.callbacks=callbacks
        self.random_state = random_state
        self.train_type = train_type
        self.attack_method = attack_method

        self.model = model

        ### Attack ####
        self.eps = eps
        self.ord = ord
        ###############

    def fit(self, X, y, sample_weight=None):
        verbose = 0 if not DEBUG else 1
        Y = self.lbl_enc.transform(y.reshape(-1, 1))
        y_ori = y
        loss_fn = get_loss(self.loss_name)

        if 'mse' in self.loss_name:
            dataset = data_utils.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        else:
            dataset = data_utils.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y))

        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=True, num_workers=2)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.train()
        for epoch in range(1, self.epochs+1):
            train_loss = 0.
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                if 'adv' in self.train_type:
                    x = projected_gradient_descent(self.model, x, y=y,
                        eps_iter=0.01, eps=self.eps, ord=self.ord, nb_iter=40)
                self.optimizer.zero_grad()
                loss = loss_fn(self.model(x), y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            print('epoch: {}/{}, train loss: {:.3f}'.format(epoch, self.epochs, train_loss))
        self.model.eval()

    def predict(self, X):
        x = torch.from_numpy(X).float()
        _, y_pred = self.model(x).max(1)
        return y_pred.numpy()

    def predict_proba(self, X):
        x = torch.from_numpy(X).float()
        return self.model(x).detach().numpy()