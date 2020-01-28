import numpy as np

from scipy.interpolate import BSpline
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

#class MLP(nn.Module):
#    """Basic MLP architecture."""
#
#    def __init__(self, n_features):
#        super(MLP, self).__init__()
#        self.hidden = nn.Linear(n_features, 512)
#        self.output = nn.Linear(512, 1)
#
#    def forward(self, x):
#        x = F.relu(self.hidden(x))
#        x = self.output(x)
#        return x

random_state = np.random.RandomState(0)
s = 10
delta = 0.
m = 1.0
sigma = 0.10 # y noise
device = 'cpu'
eps = 0.2
theta = 0.01 # regularization

n = 10000

def fs(x):
    return np.ceil(x - 0.5)

v = random_state.randn(n)
#x = random_state.rand(n) * s
#x = fs(x)
x = random_state.choice(np.arange(s), p=([1-0.1*(s-1)] + [0.1]*(s-1)), size=n).astype(np.float)

pert = random_state.binomial(1, p=delta, size=n)
x[np.where(pert)] += eps * (random_state.binomial(1, p=0.5, size=sum(pert))*2-1)
y = m * x + sigma * v
x = x.reshape(n, 1)


knots = []
for i in range(s):
    knots += [i-eps, i, i+eps]
knots = np.asarray(knots)

w = np.ones(len(knots))

def f(w):
    sp = BSpline(t=knots, c=w, k=3)
    return np.linalg.norm(sp(x).ravel() - y) + np.linalg.norm(w) * theta

def advf(w):
    sp = BSpline(t=knots, c=w, k=3)
    pn = np.asarray([sp(x-eps).ravel(), sp(x+eps).ravel(), sp(x).ravel()])
    pn = pn - y .reshape(1, -1)
    pn = np.max(pn, axis=0)

    return np.linalg.norm(pn)# + np.linalg.norm(w) * theta

#w = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
res = minimize(f, x0=w, method="L-BFGS-B")
#res = minimize(f, x0=w, method="CG")
print(res)
w = res.x

inputs = (np.arange(int((s-0.8)*100)) / 100.).reshape(int((s-0.8)*100), 1)
#output = model(torch.from_numpy(inputs).float()).detach().numpy()
print(knots)
sp = BSpline(t=knots, c=w, k=7)
output = sp(inputs)
plt.scatter(x.ravel(), y)
plt.plot(inputs, output, color="red")
plt.savefig("./temp.png")


#dataset = torch.utils.data.TensorDataset(
#    torch.from_numpy(x).float(), torch.from_numpy(y).float())
#train_loader = torch.utils.data.DataLoader(dataset,
#    batch_size=128, shuffle=True, num_workers=1)
#
#def train(model, loader, optimizer, epoch, mode="natural"):
#    model.train()
#    for batch_idx, (data, target) in enumerate(loader):
#        data, target = data.to(device), target.to(device)
#        optimizer.zero_grad()
#        output = model(data).flatten()
#        if mode == "natural":
#            loss = F.mse_loss(output, target)
#        elif mode == "adv":
#            output_p = model(data+eps).flatten()
#            output_n = model(data-eps).flatten()
#            loss = torch.max(torch.max(
#                F.mse_loss(output, target),
#                F.mse_loss(output_p, target)),
#                F.mse_loss(output_n, target))
#        loss.backward()
#        optimizer.step()
#        if epoch % 10 == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(loader.dataset),
#                100. * batch_idx / len(loader), loss.item()))
#
#model = MLP(n_features=1)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
#for i in range(500):
#    train(model, train_loader, optimizer, i, mode='adv')
