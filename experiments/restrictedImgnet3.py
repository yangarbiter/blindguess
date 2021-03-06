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
from lolip.variables import get_file_name


def load_result(auto_var):
    model_path = get_file_name(auto_var) + ".pkl"
    model_path = model_path.split("-")
    model_path[0] = 'pgd'
    model_path = '-'.join(model_path)
    model_path = os.path.join("./results/restrictedImgnet2/", model_path)
    res = joblib.load(model_path)
    ret = {
        'trn_acc': res['trn_acc'],
        'tst_acc': res['tst_acc'],
        'avg_trn_lip_1': res['avg_trn_lip_1'],
        'avg_tst_lip_1': res['avg_tst_lip_1'],
    }
    del res
    return ret

def run_restrictedImgnet_3(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = set_random_seed(auto_var)
    norm = auto_var.get_var("norm")
    trn_ds, tst_ds = auto_var.get_var("dataset", eval_trn=True)
    print(len(trn_ds), len(tst_ds))
    lbl_enc = None # TODO
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(trn_ds.classes)

    result = load_result(auto_var)
    print(f"loaded results: {result}")
    mock_trnX = np.concatenate([trn_ds[0][0], trn_ds[1][0]], axis=0)
    trny = np.array(trn_ds.targets)
    tsty = np.array(tst_ds.targets)
    result['model_path'], model = load_model(auto_var, mock_trnX, trny, None, None, 3)

    #model_path = get_file_name(auto_var).split("-")
    #ori_res = joblib.load(f"./results/restrictedImgnet/{get_file_name(auto_var).split('-')}.pkl")

    #result['trn_acc'] = (model.predict_ds(trn_ds) == trny).mean()
    result['tst_acc'] = (model.predict_ds(tst_ds) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    attack_name = auto_var.get_variable_name("attack")
    attack_model = auto_var.get_var("attack", model=model, n_classes=n_classes)
    #if 'multitarget' not in attack_name:
    #if 'multitarget' in attack_name:
    #    result['adv_trn_acc'] = np.nan
    #else:
    with Stopwatch("Attacking Train"):
        adv_trnX = attack_model.perturb_ds(trn_ds)
    result['adv_trn_acc'] = (model.predict(adv_trnX) == trny).mean()
    del adv_trnX
    with Stopwatch("Attacking Test"):
        adv_tstX = attack_model.perturb_ds(tst_ds)
    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty).mean()
    del adv_tstX
    print(f"adv trn acc: {result['adv_trn_acc']}")
    print(f"adv tst acc: {result['adv_tst_acc']}")
    del attack_model

    print(result)
    return result
