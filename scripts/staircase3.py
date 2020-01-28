from functools import partial
import numpy as np

from scipy.interpolate import BSpline, UnivariateSpline
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
import csaps

from robust_splines_sklearn import BSplineFeatures
import matplotlib.pyplot as plt

random_state = np.random.RandomState(1)
s = 10
s0 = 5
delta = 0.05
m = 1.0
sigma = 0.10 # y noise
eps = 0.2
#theta = 0.00001 # regularization
theta = 0.3 # regularization
#theta = 0.1 # regularization
beta = 1

def _get_natural_f(knots):
    """Returns mapping of natural cubic spline values to 2nd derivatives.
    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, pp 145-146
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: A 2-d array mapping natural cubic spline values at
     knots to second derivatives.
    :raise ImportError: if scipy is not found, required for
     ``linalg.solve_banded()``
    """
    try:
        from scipy import linalg
    except ImportError: # pragma: no cover
        raise ImportError("Cubic spline functionality requires scipy.")

    h = knots[1:] - knots[:-1]
    diag = (h[:-1] + h[1:]) / 3.
    ul_diag = h[1:-1] / 6.
    banded_b = np.array([np.r_[0., ul_diag], diag, np.r_[ul_diag, 0.]])
    d = np.zeros((knots.size - 2, knots.size))
    for i in range(knots.size - 2):
        d[i, i] = 1. / h[i]
        d[i, i + 2] = 1. / h[i + 1]
        d[i, i + 1] = - d[i, i] - d[i, i + 2]

    fm = linalg.solve_banded((1, 1), banded_b, d)

    return np.vstack([np.zeros(knots.size), fm, np.zeros(knots.size)])


def fs(x):
    return np.ceil(x - 0.5)

def gen_data(n):
    v = random_state.randn(n)
    x = random_state.choice(np.arange(s), p=(
        #[1/s0 - 0.05*(s-s0)/s0] * s0 + [0.05] * (s-s0)), size=n).astype(np.float)
        [1/s0 - 0.01*(s-s0)/s0] * s0 + [0.01] * (s-s0)), size=n).astype(np.float)

    pert = random_state.binomial(1, p=delta, size=n)
    x[np.where(pert)] += eps * (random_state.binomial(1, p=0.5, size=sum(pert))*2-1)
    y = fs(m * x) + sigma * v
    x = x.reshape(n, 1)
    return x, y

x, y = gen_data(n=40)
tstx, tsty = gen_data(n=1000)
tsty = m*fs(tstx).ravel()

#knots = [-1]
knots = []
knoty = []
for i in range(s):
    knots += [i-eps, i, i+eps]
    #knoty += [m*i, m*i, m*i]
    knoty += [m*(i-eps), m*i, m*(i-eps)]
#knots += [s+1]
knots = np.asarray(knots)
knoty = np.asarray(knoty)

Omega = 1. / (np.diag(knoty)+1e-6)
#Omega = _get_natural_f(knots)
#import ipdb; ipdb.set_trace()
#Omega
#Omege = np.dot(Omega, Omega.T)

fet_transformer = BSplineFeatures(knots=knots, degree=3, periodic=False)
X = fet_transformer.fit_transform(x)
X = X * x * m
tstX = fet_transformer.transform(tstx)
tstX = tstX * tstx * m
#X = np.hstack((X, np.ones((n, 1))))
print(X.shape)

def predict(w, X):
    return np.dot(X, w)

def regularization(w, theta):
    #return theta * np.dot(np.dot(Omega, w[:, np.newaxis])[:, 0], w)
    #return np.linalg.norm(predict(w, X) - y) * theta
    #return np.linalg.norm(w - knoty) * theta
    #return np.abs(w - knoty) * theta
    return np.linalg.norm(w) * theta

def f(w, X, y, theta):
    #return np.sqrt((predict(w, X) - y)**2).mean() + np.linalg.norm(w) * theta
    #return np.abs((predict(w, X) - y)).mean() + np.linalg.norm(w) * theta
    return np.abs((predict(w, X) - y)).mean() + regularization(w, theta)

def advf(w, X, y, theta):
    pn = np.asarray([predict(w, X), predict(w, X+eps), predict(w, X-eps)])
    #pn = np.sqrt((pn - y)**2).max(0).reshape(-1)
    pn = np.abs(pn - y).max(0)
    #return pn.mean() + np.linalg.norm(w) * theta
    return pn.mean() + regularization(w, theta)

def trades(w, X, y, theta, beta):
    pred = predict(w, X)
    loss = np.abs((pred - y)).mean()
    pn = np.asarray([pred, predict(w, X+eps), predict(w, X-eps)])
    pn = np.abs(pn - pred).max(0)
    return loss + beta * pn.mean() + regularization(w, theta)

#res = minimize(trades, x0=w, method="L-BFGS-B")
#w = np.ones(3*s+2)
#res = minimize(uspf, x0=w, method="L-BFGS-B")
#spw = res.x

def cv_best(obj_fn, w0):
    ws = []
    ts = []
    tst_objs = []
    #for t in [0., 1e-4, 1e-6, 1e-8, 1e-10]:
    for t in [0.]:
    #for t in [1, 0., 1e-1, 1e-2, 1e1, 1e2]:
        res = minimize(obj_fn, x0=w0, method="L-BFGS-B", args=(X, y, t,))
        ws.append(res.x)
        ts.append(t)
        tst_objs.append(f(ws[-1], tstX, tsty, theta=0))
    print(tst_objs)
    return ws[np.argmin(tst_objs)], ts[np.argmin(tst_objs)]


w = random_state.rand(X.shape[1])
natw, natt = cv_best(f, w)
print(natt)

advw, advt = cv_best(advf, w)
print(advt)

tradesw, tradest = cv_best(partial(trades, beta=1.0), w)
print(tradest)

reg = LinearRegression(fit_intercept=False)
reg.fit(X, y)

import ipdb; ipdb.set_trace()

inputs = (np.arange(int((s-0.8)*100)) / 100.).reshape(int((s-0.8)*100), 1)
tX = fet_transformer.transform(inputs)
tX = tX * inputs * m
plt.scatter(x.ravel(), y)
#plt.scatter(tstx.ravel(), tsty)

for i in range(tX.shape[1]):
    plt.plot(inputs, tX[:, i], color="blue")
#plt.plot(inputs, tX[:, 28], label="2")

#sp = csaps.UnivariateCubicSmoothingSpline(knots.ravel(), knoty, smooth=0.1)
#sp = UnivariateSpline(x=knots, y=m*fs(knots.ravel()), w=w, k=3, s=theta)
#plt.plot(inputs, sp(inputs.ravel()), color="gray", label="spsmooth")
#plt.plot(inputs, usp_predict(spw, inputs), color="blue", label="spnat")
plt.plot(inputs, reg.predict(tX), color="black", label="lin")
plt.plot(inputs, predict(natw, tX), color="red", label="nat")
plt.plot(inputs, predict(advw, tX), color="green", label="adv")
plt.plot(inputs, predict(tradesw, tX), color="orange", label="trades b=1")
#tw = np.ones(30)
#tw[1] = 10
#tw[2] = 10
#tw[3] = 10
#plt.plot(inputs, usp_predict(tw, inputs, theta=0.1), color="black", label="nat")
plt.legend()

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
