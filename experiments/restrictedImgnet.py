import os

import torch
from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import set_random_seed
from lolip.utils import estimate_local_lip_v2
from lolip.variables import get_file_name


def calc_lip(model, X, Xp, top_norm, btm_norm):
    top = np.linalg.norm(model.predict_real(X)-model.predict_real(Xp), ord=top_norm, axis=1)
    down = np.linalg.norm(X.reshape(len(Xp), -1)-Xp.reshape(len(Xp), -1), ord=btm_norm, axis=1)
    return np.mean(top / (down+1e-6))

def run_restrictedImgnet(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = set_random_seed(auto_var)
    norm = auto_var.get_var("norm")
    trn_ds, tst_ds = auto_var.get_var("dataset")
    lbl_enc = None # TODO
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(trn_ds.classes)

    result = {}
    mock_trnX = np.concatenate([trn_ds[0][0], trn_ds[1][0]], axis=0)
    mock_trny = np.arange(n_classes)
    multigpu = True if torch.cuda.device_count() > 1 else False
    model = auto_var.get_var("model", trnX=mock_trnX, trny=mock_trny,
                             multigpu=multigpu, n_channels=3)
    model.tst_ds = tst_ds
    result['model_path'] = os.path.join('./models', get_file_name(auto_var) + '-ep%04d.pt')
    if None:
        result['model_path'] = result['model_path'] % model.epochs
        model.load(result['model_path'])
        model.model.to(device)
    else:
        with Stopwatch("Fitting Model"):
            history = model.fit_dataset(trn_ds)
        model.save(result['model_path'])
        result['model_path'] = result['model_path'] % model.epochs
        result['history'] = history

    result['trn_acc'] = np.nan
    result['tst_acc'] = (model.predict_ds(tst_ds) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    attack_model = auto_var.get_var("attack", model=model, n_classes=n_classes)
    with Stopwatch("Attacking"):
        #adv_trnX = attack_model.perturb(trnX, trny)
        adv_tstX = attack_model.perturb(tstX, tsty)
    result['adv_trn_acc'] = np.nan
    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty).mean()
    print(f"adv trn acc: {result['adv_trn_acc']}")
    print(f"adv tst acc: {result['adv_tst_acc']}")
    del attack_model

    result['avg_trn_lip'] = np.nan
    with Stopwatch("Estimating tst Lip"):
        _, tst_lip = estimate_local_lip_v2(model.model, tstX, top_norm=2, btm_norm=norm,
                                     epsilon=auto_var.get_var("eps"), device=device)
    result['avg_tst_lip'] = calc_lip(model, tstX, tst_lip, top_norm=2, btm_norm=norm).mean()
    print(f"avg trn lip: {result['avg_trn_lip']}")
    print(f"avg tst lip: {result['avg_tst_lip']}")

    print(result)
    return result
