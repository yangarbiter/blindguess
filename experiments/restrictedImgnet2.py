from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

import torch
from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from .utils import set_random_seed
from .experiment02 import load_model
from lolip.utils import estimate_local_lip_v2
from lolip.variables import get_file_name


def run_restrictedImgnet_2(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = set_random_seed(auto_var)
    norm = auto_var.get_var("norm")
    trn_ds, tst_ds = auto_var.get_var("dataset")
    print(len(trn_ds), len(tst_ds))
    lbl_enc = None # TODO
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(trn_ds.classes)

    result = {}
    mock_trnX = np.concatenate([trn_ds[0][0], trn_ds[1][0]], axis=0)
    trny = np.array(trn_ds.targets)
    tsty = np.array(tst_ds.targets)
    result['model_path'], model = load_model(auto_var, mock_trnX, trny, None, None, 3)

    #model_path = get_file_name(auto_var).split("-")
    #ori_res = joblib.load(f"./results/restrictedImgnet/{get_file_name(auto_var).split('-')}.pkl")

    result['trn_acc'] = (model.predict_ds(trn_ds) == trny).mean()
    result['tst_acc'] = (model.predict_ds(tst_ds) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    attack_model = auto_var.get_var("attack", model=model, n_classes=n_classes)
    with Stopwatch("Attacking"):
        adv_trnX = attack_model.perturb_ds(trn_ds)
        adv_tstX = attack_model.perturb_ds(tst_ds)
    result['adv_trn_acc'] = (model.predict(adv_trnX) == trny).mean()
    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty).mean()
    print(f"adv trn acc: {result['adv_trn_acc']}")
    print(f"adv tst acc: {result['adv_tst_acc']}")
    del attack_model

    with Stopwatch("Estimating tst Lip"):
        trn_lip, _ = estimate_local_lip_v2(model.model, trn_ds, top_norm=1, btm_norm=norm,
                                     epsilon=auto_var.get_var("eps"), device=device)
        tst_lip, _ = estimate_local_lip_v2(model.model, tst_ds, top_norm=1, btm_norm=norm,
                                     epsilon=auto_var.get_var("eps"), device=device)
    result['avg_trn_lip_1'] = trn_lip
    result['avg_tst_lip_1'] = tst_lip
    print(f"avg trn lip: {result['avg_trn_lip_1']}")
    print(f"avg tst lip: {result['avg_tst_lip_1']}")

    print(result)
    return result
