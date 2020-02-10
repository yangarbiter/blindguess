import os

import torch
from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from .utils import set_random_seed
from lolip.utils import estimate_local_lip_v2
from lolip.variables import get_file_name

from .utils import load_model


def load_result(auto_var):
    model_path = get_file_name(auto_var) + ".pkl"
    model_path = model_path.split("-")
    model_path[0] = 'pgd'
    model_path = '-'.join(model_path)
    model_path = os.path.join("./results/experiment03", model_path)
    res = joblib.load(model_path)
    ret = {
        'trn_acc': res['trn_acc'],
        'tst_acc': res['tst_acc'],
        'avg_trn_lip_1': res['avg_trn_lip_1'],
        'avg_tst_lip_1': res['avg_tst_lip_1'],
        'avg_trn_lip_kl': res['avg_trn_lip_kl'],
        'avg_tst_lip_kl': res['avg_tst_lip_kl'],
    }
    del res
    return ret

def get_proba_list(pred, y, bin_counts=20):
    pred_proba = pred.max(axis=1)
    pred_y = pred.argmax(axis=1)
    acc_counts = np.histogram(pred_proba, bins=np.arange(bin_counts+1)/bin_counts)[0]
    acc_list = []
    std_list = []
    for i in range(bin_counts):
        proba_range = [i/bin_counts, (i+1)/bin_counts]
        idx = np.where(np.logical_and(pred_proba >= proba_range[0], pred_proba < proba_range[1]))[0]
        if len(idx) > 0:
            acc_list.append((pred_y[idx] == y[idx]).mean())
            std_list.append((pred_y[idx] == y[idx]).std())
        else:
            acc_list.append(0.)
            std_list.append(0.)
    return np.array(acc_counts), np.array(acc_list), np.array(std_list)

def brier_scores(proba_pred, y):
    hard_pred = proba_pred.argmax(1)
    scores = []
    for i in range(len(y)):
        scores.append((proba_pred[i][y[i]] - (hard_pred[i] == y[i]))**2)
    return np.array(scores)

def calibrate_abs(proba_counts, proba_list):
    bin_counts = len(proba_list)
    ideal = np.arange(bin_counts) / bin_counts + 1 / 2 / bin_counts
    ret = 0.
    for i in range(bin_counts):
        ret += np.abs(ideal[i] - proba_list[i]) * proba_counts[i]

    return ret / np.sum(proba_counts)

def run_calibration(auto_var):
    bin_counts = 20
    _ = set_random_seed(auto_var)
    trnX, trny, tstX, tsty = auto_var.get_var("dataset")
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(np.unique(trny))
    n_channels = trnX.shape[-1]

    result = load_result(auto_var)
    print(f"loaded results: {result}")
    result['model_path'], model = load_model(auto_var, trnX, trny, tstX, tsty, n_channels)

    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    trn_pred = model.predict_proba(trnX)
    tst_pred = model.predict_proba(tstX)
    trn_proba_counts, trn_proba_list, trn_proba_std = get_proba_list(trn_pred, trny, bin_counts=bin_counts)
    tst_proba_counts, tst_proba_list, tst_proba_std = get_proba_list(tst_pred, tsty, bin_counts=bin_counts)

    result['avg_trn_brier_score'] = brier_scores(trn_pred, trny).mean()
    result['avg_tst_brier_score'] = brier_scores(tst_pred, tsty).mean()
    print(f"avg trn brier: {result['avg_trn_brier_score']}")
    print(f"avg tst brier: {result['avg_tst_brier_score']}")

    result['avg_trn_abs'] = calibrate_abs(trn_proba_counts, trn_proba_list)
    result['avg_tst_abs'] = calibrate_abs(tst_proba_counts, tst_proba_list)
    print(f"avg trn abs: {result['avg_trn_abs']}")
    print(f"avg tst abs: {result['avg_tst_abs']}")

    print(result)
    return result
