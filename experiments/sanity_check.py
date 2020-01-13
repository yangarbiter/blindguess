import os

import torch
from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import set_random_seed
from lolip.utils import estimate_local_mse
from lolip.variables import get_file_name
from .experiment02 import load_model


def run_sanity(auto_var):
    random_state = set_random_seed(auto_var)
    norm = auto_var.get_var("norm")
    trnX, trny, tstX, tsty = auto_var.get_var("dataset")
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(np.unique(trny))
    n_channels = trnX.shape[-1]

    n_samples = 100
    trn_idx = np.arange(len(trnX))
    trn_idx = random_state.choice(trn_idx, size=n_samples, replace=False)
    tst_idx = np.arange(len(tstX))
    tst_idx = random_state.choice(tst_idx, size=n_samples, replace=False)

    result = {}
    result['model_path'], model = load_model(auto_var, trnX, trny, tstX, tsty, n_channels=n_channels)

    result['trn_acc'] = (model.predict(trnX) == trny).mean()
    result['tst_acc'] = (model.predict(tstX) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    attack_model = auto_var.get_var("attack", model=model, n_classes=n_classes)
    with Stopwatch("Attacking"):
        adv_trnX = attack_model.perturb(trnX[trn_idx], trny[trn_idx])
        adv_tstX = attack_model.perturb(tstX[tst_idx], tsty[tst_idx])
    result['adv_trn_acc'] = (model.predict(adv_trnX) == trny[trn_idx]).mean()
    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty[tst_idx]).mean()
    print(f"adv train acc: {result['adv_trn_acc']}")
    print(f"adv test acc: {result['adv_tst_acc']}")

    with Stopwatch("Estimating trn MSE"):
        trn_lip_mse, _ = estimate_local_mse(model.model, trnX[trn_idx], epsilon=auto_var.get_var("eps"))
        result['avg_trn_mse'] = trn_lip_mse
    with Stopwatch("Estimating tst MSE"):
        tst_lip_mse, _ = estimate_local_mse(model.model, tstX[tst_idx], epsilon=auto_var.get_var("eps"))
        result['avg_tst_mse'] = tst_lip_mse
    print(f"avg trn MSE: {result['avg_trn_mse']}")
    print(f"avg tst MSE: {result['avg_tst_mse']}")

    print(result)
    return result
