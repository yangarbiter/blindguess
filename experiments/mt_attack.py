import os

import torch
from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from .utils import set_random_seed
from lolip.utils import estimate_local_lip_v2
from lolip.variables import get_file_name
from .experiment02 import load_model


def load_result(auto_var, result_dir="./results/experiment03"):
    model_path = get_file_name(auto_var) + ".pkl"
    model_path = os.path.join(result_dir, model_path)
    res = joblib.load(model_path)
    ret = {
        'trn_acc': res['trn_acc'],
        'tst_acc': res['tst_acc'],
        'avg_trn_lip_1': res['avg_trn_lip_1'],
        'avg_tst_lip_1': res['avg_tst_lip_1'],
    }
    del res
    return ret

def run_mtattack(auto_var):
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

    attack_model = auto_var.get_var("attack", model=model,
            n_classes=n_classes, clip_min=0, clip_max=1)
    with Stopwatch("Attacking"):
        adv_trnX = attack_model.perturb(trnX, trny)
        adv_tstX = attack_model.perturb(tstX, tsty)
    result['adv_trn_acc'] = (model.predict(adv_trnX) == trny).mean()
    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty).mean()
    print(f"adv trn acc: {result['adv_trn_acc']}")
    print(f"adv tst acc: {result['adv_tst_acc']}")

    #with Stopwatch("Estimating trn Lip"):
    #    trn_lip = estimate_local_lip_v2(model.model, trnX, top_norm=2, btm_norm=norm,
    #                                 epsilon=auto_var.get_var("eps"))
    #result['avg_trn_lip'] = calc_lip(model, trnX, trn_lip, top_norm=2, btm_norm=norm).mean()
    #with Stopwatch("Estimating tst Lip"):
    #    tst_lip = estimate_local_lip_v2(model.model, tstX, top_norm=2, btm_norm=norm,
    #                                 epsilon=auto_var.get_var("eps"))
    #result['avg_tst_lip'] = calc_lip(model, tstX, tst_lip, top_norm=2, btm_norm=norm).mean()
    #print(f"avg trn lip: {result['avg_trn_lip']}")
    #print(f"avg tst lip: {result['avg_tst_lip']}")

    print(result)
    return result
