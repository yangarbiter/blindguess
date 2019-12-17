
import torch.nn as nn
import torch.nn.functional as F

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