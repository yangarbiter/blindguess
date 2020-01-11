import os

import torch
from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import set_random_seed
from lolip.utils import estimate_local_lip_v2
from lolip.variables import get_file_name


def load_model(auto_var, trnX, trny, tstX, tsty, n_channels):
    model = auto_var.get_var("model", trnX=trnX, trny=trny, n_channels=n_channels)
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


def run_experiment02(auto_var):
    _ = set_random_seed(auto_var)
    #norm = auto_var.get_var("norm")
    trnX, trny, tstX, tsty = auto_var.get_var("dataset")
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(np.unique(trny))
    n_channels = trnX.shape[-1]

    result = {}
    result['model_path'], model = load_model(auto_var, trnX, trny, tstX, tsty, n_channels)

    result['trn_acc'] = (model.predict(trnX) == trny).mean()
    result['tst_acc'] = (model.predict(tstX) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    attack_model = auto_var.get_var("attack", model=model, n_classes=n_classes)
    with Stopwatch("Attacking"):
        if len(trnX) < 90000:
            adv_trnX = attack_model.perturb(trnX, trny)
        adv_tstX = attack_model.perturb(tstX, tsty)
    if len(trnX) < 90000:
        result['adv_trn_acc'] = (model.predict(adv_trnX) == trny).mean()
    else:
        result['adv_trn_acc'] = np.nan
    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty).mean()
    print(f"adv trn acc: {result['adv_trn_acc']}")
    print(f"adv tst acc: {result['adv_tst_acc']}")
    del attack_model

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
