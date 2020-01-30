import os

import torch
from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import set_random_seed
from lolip.utils import estimate_local_lip_v2
from lolip.variables import get_file_name
from .experiment02 import load_model


def run_experiment03(auto_var):
    _ = set_random_seed(auto_var)
    norm = auto_var.get_var("norm")
    trnX, trny, tstX, tsty = auto_var.get_var("dataset")
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(np.unique(trny))
    n_channels = trnX.shape[-1]

    result = {}
    result['model_path'], model = load_model(auto_var, trnX, trny, tstX, tsty, n_channels=n_channels)

    #model = auto_var.get_var("model", trnX=trnX, trny=trny)
    #model.tst_ds = (tstX, tsty)
    #model_path = get_file_name(auto_var).split("-")
    #model_path[0] = 'pgd'
    #model_path = '-'.join(model_path)
    #model_path = os.path.join('./models', model_path + '.pt')
    #result['model_path'] = model_path
    #model.load(result['model_path'])
    #model.model.cuda()

    result['trn_acc'] = (model.predict(trnX) == trny).mean()
    result['tst_acc'] = (model.predict(tstX) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    if len(trnX) < 90000:
        with Stopwatch("Estimating trn Lip (L1)"):
            trn_lip_1, _ = estimate_local_lip_v2(model.model, trnX, top_norm=1, btm_norm=norm,
                                         epsilon=auto_var.get_var("eps"))
            result['trn_lip_1_pert'] = np.nan
            result['avg_trn_lip_1'] = trn_lip_1
    else:
        result['trn_lip_1_pert'] = np.nan
        result['avg_trn_lip_1'] = np.nan
    with Stopwatch("Estimating tst Lip (L1)"):
        tst_lip_1, _ = estimate_local_lip_v2(model.model, tstX, top_norm=1, btm_norm=norm,
                                     epsilon=auto_var.get_var("eps"))
        result['tst_lip_1_pert'] = np.nan
        result['avg_tst_lip_1'] = tst_lip_1
    print(f"avg trn lip (L1): {result['avg_trn_lip_1']}")
    print(f"avg tst lip (L1): {result['avg_tst_lip_1']}")

    if len(trnX) < 90000:
        with Stopwatch("Estimating trn Lip (KL)"):
            trn_lip_kl, _ = estimate_local_lip_v2(model.model, trnX, top_norm='kl', btm_norm=norm,
                                         epsilon=auto_var.get_var("eps"))
            result['trn_lip_kl_pert'] = np.nan
            result['avg_trn_lip_kl'] = trn_lip_kl
    else:
        result['trn_lip_kl_pert'] = np.nan
        result['avg_trn_lip_kl'] = np.nan
    with Stopwatch("Estimating tst Lip (KL)"):
        tst_lip_kl, _ = estimate_local_lip_v2(model.model, tstX, top_norm='kl', btm_norm=norm,
                                     epsilon=auto_var.get_var("eps"))
        result['tst_lip_kl_pert'] = np.nan
        result['avg_tst_lip_kl'] = tst_lip_kl
    print(f"avg trn lip (KL): {result['avg_trn_lip_kl']}")
    print(f"avg tst lip (KL): {result['avg_tst_lip_kl']}")

    print(result)
    return result
