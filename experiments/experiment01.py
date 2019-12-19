from functools import partial

from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from .utils import set_random_seed
from lolip.utils import estimate_local_lip


def calc_lip(model, X, Xp):
    top = np.linalg.norm(model.predict_real(X)-model.predict_real(Xp), axis=1)
    down = np.linalg.norm(X.reshape(len(Xp), -1)-Xp.reshape(len(Xp), -1), axis=1)
    return np.mean(top / (down+1e-6))

def run_experiment01(auto_var):
    #random_state = set_random_seed(auto_var)
    norm = auto_var.get_var("norm")
    trnX, trny, tstX, tsty = auto_var.get_var("dataset")
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)

    result = {}
    model = auto_var.get_var("model", trnX=trnX, trny=trny)
    model.tst_ds = (tstX, tsty)
    with Stopwatch("Fitting Model"):
        model.fit(trnX, trny)

    result['trn_acc'] = (model.predict(trnX) == trny).mean()
    result['tst_acc'] = (model.predict(tstX) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    attack_model = auto_var.get_var("attack", model=model)
    with Stopwatch("Attacking"):
        adv_trnX = attack_model.perturb(trnX)
        adv_tstX = attack_model.perturb(tstX)
    result['adv_trn_acc'] = (model.predict(adv_trnX) == trny).mean()
    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty).mean()
    print(f"adv trn acc: {result['adv_trn_acc']}")
    print(f"adv tst acc: {result['adv_tst_acc']}")

    with Stopwatch("Estimating trn Lip"):
        trn_lip = estimate_local_lip(model.model, trnX,
                                     norm=norm, epsilon=auto_var.get_var("eps"))
    result['avg_trn_lip'] = calc_lip(model, trnX, trn_lip).mean()

    with Stopwatch("Estimating tst Lip"):
        tst_lip = estimate_local_lip(model.model, tstX,
                                     norm=norm, epsilon=auto_var.get_var("eps"))
    result['avg_tst_lip'] = calc_lip(model, tstX, tst_lip).mean()

    #import torch
    #from lolip.utils import local_lip
    #local_lip(model.model, torch.from_numpy(tstX.transpose(0, 3, 1, 2)).cuda().float(), torch.from_numpy(tst_lip.transpose(0, 3, 1, 2)).cuda().float())


    return result
