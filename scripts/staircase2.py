from functools import partial
import numpy as np

from scipy.interpolate import BSpline, UnivariateSpline
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import csaps

from robust_splines_sklearn import BSplineFeatures
import matplotlib.pyplot as plt

random_state = np.random.RandomState(1)
s = 10
s0 = 5
delta = 0.05
m = 1.0
sigma = 0.15 # y noise
eps = 0.25
#theta = 0.00001 # regularization
theta = 0.3 # regularization
#theta = 0.1 # regularization
beta = 1

def fs(x):
    return np.ceil(x - 0.5)

def gen_data(n):
    v = random_state.randn(n)
    x = random_state.choice(np.arange(s), p=(
        [1/s0 - 0.01*(s-s0)/s0] * s0 + [0.01] * (s-s0)), size=n).astype(np.float)

    pert = random_state.binomial(1, p=delta, size=n)
    x[np.where(pert)] += eps * (random_state.binomial(1, p=0.5, size=sum(pert))*2-1)
    y = fs(m * x) + sigma * v
    x = x.reshape(n, 1)
    return x, y

x, y = gen_data(n=1000)
tstx, tsty = gen_data(n=1000)
#tsty = m*fs(tstx).ravel()

x, y = zip(*sorted(zip(x, y)))
x, y = np.asarray(x), np.asarray(y)

#knots = [-1]
knots = []
knoty = []
for i in range(s):
    knots += [i-eps, i, i+eps]
    #knoty += [m*i, m*i, m*i]
    knoty += [m*(i-eps), m*i, m*(i+eps)]
#knots += [s+1]
knots = np.asarray(knots)
knoty = np.asarray(knoty)


fet_transformer = BSplineFeatures(knots=knots, degree=3, periodic=True)
X = fet_transformer.fit_transform(x)
#X = np.hstack((X, np.ones((n, 1))))
print(X.shape)

def predict(w, X):
    return np.dot(X, w)

def sp_predict(w, X):
    sp = BSpline(t=knots, c=w, k=3)
    return sp(X).ravel()

def usp_predict(w, X, theta=theta):
    sp = UnivariateSpline(x=knots, y=knoty, w=w, k=3, s=theta)
    #sp = BSpline(t=knots, c=w, k=3)
    return sp(X).ravel()
    
def uspf(w, x, y, theta=0.1):
    return np.abs((usp_predict(w, x, theta) - y)).mean() + np.linalg.norm(w) * theta

def advspf(w, x, y, theta=0.1):
    pn = np.asarray([
        usp_predict(w, x, theta), usp_predict(w, x+eps, theta), usp_predict(w, x-eps, theta)])
    pn = np.abs(pn - y).max(0)
    return pn.mean() + np.linalg.norm(w) * theta

def tradespf(w, x, y, theta=0.1, beta=1.):
    pred = usp_predict(w, x, theta)
    loss = np.abs((pred - y)).mean()
    pn = np.asarray([pred, usp_predict(w, x+eps, theta), usp_predict(w, x-eps, theta)])
    pn = np.abs(pn - pred).max(0)
    return loss + beta * pn.mean() + np.linalg.norm(w) * theta

#def spf(w, theta=0.1):
#    #return np.sqrt((predict(w, X) - y)**2).mean() + np.linalg.norm(w) * theta
#    return np.abs((sp_predict(w, x, theta) - y)).mean()

#def f(w):
#    #return np.sqrt((predict(w, X) - y)**2).mean() + np.linalg.norm(w) * theta
#    return np.abs((predict(w, X) - y)).mean() + np.linalg.norm(w) * theta
#
#def advf(w):
#    pn = np.asarray([predict(w, X), predict(w, X+eps), predict(w, X-eps)])
#    #pn = np.sqrt((pn - y)**2).max(0).reshape(-1)
#    pn = np.abs(pn - y).max(0)
#    return pn.mean() + np.linalg.norm(w) * theta
#
#def trades(w):
#    pred = predict(w, X)
#    loss = np.abs((pred - y)).mean()
#    pn = np.asarray([pred, predict(w, X+eps), predict(w, X-eps)])
#    pn = np.abs(pn - pred).max(0)
#    return loss + beta * pn.mean() + np.linalg.norm(w) * theta

#res = minimize(trades, x0=w, method="L-BFGS-B")
#w = np.ones(3*s+2)
#res = minimize(uspf, x0=w, method="L-BFGS-B")
#spw = res.x

def cv_best(obj_fn, w0):
    ws = []
    ts = []
    tst_objs = []
    for t in [0, 1, 1e2, 1e4, 1e6, 1e8]:
        res = minimize(obj_fn, x0=w0, method="L-BFGS-B", args=(x, y, t,))
        ws.append(res.x)
        ts.append(t)
        tst_objs.append(uspf(ws[-1], tstx, tsty, theta=0))
    print(tst_objs)
    return ws[np.argmin(tst_objs)], ts[np.argmin(tst_objs)]


w = random_state.rand(X.shape[1])
w = np.ones(X.shape[1])
natw, natt = cv_best(uspf, w)
print(natt, natw)

advw, advt = cv_best(advspf, w)
print(advt, advw)

tradesw, tradest = cv_best(partial(tradespf, beta=1.0), w)
print(tradest, tradesw)

inputs = (np.arange(int((s-0.8)*100)) / 100.).reshape(int((s-0.8)*100), 1)
tX = fet_transformer.transform(inputs)
plt.scatter(x.ravel(), y)
#plt.scatter(tstx.ravel(), tsty)

#sp = csaps.UnivariateCubicSmoothingSpline(knots.ravel(), knoty, smooth=0.1)
#sp = UnivariateSpline(x=knots, y=m*fs(knots.ravel()), w=w, k=3, s=theta)
#plt.plot(inputs, sp(inputs.ravel()), color="gray", label="spsmooth")
#plt.plot(inputs, usp_predict(spw, inputs), color="blue", label="spnat")
plt.plot(inputs, usp_predict(natw, inputs, theta=natt), color="red", label="nat")
#plt.plot(inputs, usp_predict(natw, inputs, theta=10000), color="black", label="nat10000")
#plt.plot(inputs, usp_predict(natw, inputs, theta=1), color="black", label="nat1")
plt.plot(inputs, usp_predict(advw, inputs, theta=advt), color="green", label="adv")
plt.plot(inputs, usp_predict(tradesw, inputs, theta=tradest), color="orange", label="trades b=1")
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
