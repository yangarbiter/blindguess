import os

import torch
from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from .utils import set_random_seed
from lolip.utils import estimate_local_lip_v2
from lolip.variables import get_file_name


def load_model(auto_var, trnX, trny, tstX, tsty, n_channels):
    model = auto_var.get_var("model", trnX=trnX, trny=trny, n_channels=n_channels)
    if tstX is not None and tsty is not None:
        model.tst_ds = (tstX, tsty)
    model_path = get_file_name(auto_var).split("-")
    model_path[0] = 'pgd'
    model_path = '-'.join(model_path)
    if os.path.exists(os.path.join('./models', model_path + '-ep%04d.pt') % model.epochs):
        model_path = os.path.join('./models', model_path + '-ep%04d.pt') % model.epochs
    else:
        model_path = os.path.join('./models', model_path + '.pt')
    #model_path = os.path.join('./models', model_path + '-ep%04d.pt') % 2

    model.load(model_path)
    model.model.cuda()
    return model_path, model

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

def run_experiment02(auto_var):
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
    attack_name = auto_var.get_variable_name("attack")
    attack_model = auto_var.get_var("attack", model=model,
            n_classes=n_classes, clip_min=0, clip_max=1)
    with Stopwatch("Attacking"):
        if 'multitarget' not in attack_name:
            adv_trnX = attack_model.perturb(trnX, trny)
        adv_tstX = attack_model.perturb(tstX, tsty)
    if 'multitarget' in attack_name:
        result['adv_trn_acc'] = np.nan
    else:
        result['adv_trn_acc'] = (model.predict(adv_trnX) == trny).mean()
    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty).mean()
    print(f"adv trn acc: {result['adv_trn_acc']}")
    print(f"adv tst acc: {result['adv_tst_acc']}")

    print(result)
    return result
