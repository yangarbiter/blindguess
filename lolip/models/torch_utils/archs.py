from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, wide_resnet101_2

from .wideresnet import *
from .resnet import resnet50

def tWRN50_2(n_classes):
    return wide_resnet50_2(num_classes=n_classes)

def tWRN101_2(n_classes):
    return wide_resnet50_2(num_classes=n_classes)

def ResNet50(n_classes):
    return resnet50(pretrained=False, n_channels=1, num_classes=n_classes)

class CNN001(nn.Module):
    def __init__(self, n_classes):
        super(CNN001, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x

class CNN002(nn.Module):
    """https://github.com/yaodongyu/TRADES/blob/e20f7b9b99c79ed3cf0d1bb12a47c229ebcac24a/models/small_cnn.py#L5"""
    def __init__(self, n_classes, drop=0.5):
        super(CNN002, self).__init__()

        self.num_channels = 1

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, n_classes)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

class MLP(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(n_features, 256)
        self.logits = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.softmax(self.logits(x), dim=1)
        return x

class realMLP(nn.Module):

    def __init__(self, n_features, n_classes):
        super(realMLP, self).__init__()
        self.hidden = nn.Linear(n_features, 256)
        self.output = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x
