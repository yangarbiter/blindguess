from collections import namedtuple

import numpy as np

from stair_utils import get_P, data_gen, label_noise, generate_x_noise
from stair_utils import gen_feats, gen_test_set
from stair_utils import solve_normal, solve_adv, solve_trades

np.random.seed(0)

num_stairs = 10
num_examples = 10
adv_eps = (1.0 / 3)
noise_eps = 0.0
x_noise = 0.0
slope = 1
#theta = np.random.randn(feats.shape[1])
lamda = 0.01
np.set_printoptions(precision=5)

knots = np.r_[np.arange(num_stairs), np.arange(num_stairs)-adv_eps, np.arange(num_stairs)+adv_eps]
knots = np.sort(knots)

# weights on different stairs
weights_1 = np.asarray([1/5]*5)
weights_2 = np.asarray([0.01]*(num_stairs-5))
weights = np.concatenate([weights_1, weights_2])
weights /= np.sum(weights)

Result = namedtuple('Result', ['x'])

P = get_P(knots)

num_examples = 40
X = data_gen(num_examples, weights)
y = slope*X + label_noise(X.shape[0], noise_eps)
X += generate_x_noise(num_examples, x_noise, adv_eps)

X_test, y_test = get_test_set(num_stairs, slope, adv_eps)
    
feats = get_feats(X, knots)
print(feats.shape)
res_normal = solve_normal(feats, y, lamda)

# adversarial training data
T_feats = [get_feats(xx, knots) for xx in [X-adv_eps, X, X+adv_eps]]

res_adv = solve_adv(T_feats, y, lamda, P)
res_trd = solve_trades(T_feats, y, lamda, P, beta=1)

X_plot = np.linspace(-adv_eps, num_stairs-1 + adv_eps, 100)
feats_plot = get_feats(X_plot, knots)
normal_plot_preds = feats_plot.dot(res_normal.x)

plt.scatter(X, y)
plt.plot(X_plot, normal_plot_preds, label="Normal")
adv_plot_preds = feats_plot.dot(res_adv.x)
plt.plot(X_plot, adv_plot_preds, label="Adversarial")
trd_plot_preds = feats_plot.dot(res_trd.x)
plt.plot(X_plot, trd_plot_preds, label="Trades")
plt.legend()
print("Normal")
summarize(res_normal, X)
print()
print("Adversarial")
summarize(res_adv, X)
print()
print("Trades")
summarize(res_trd, X)