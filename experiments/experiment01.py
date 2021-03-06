import os
import logging

import torch
from bistiming import Stopwatch
from mkdir_p import mkdir_p
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import set_random_seed
from lolip.utils import estimate_local_lip_v2
from lolip.variables import get_file_name

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def run_experiment01(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = set_random_seed(auto_var)
    norm = auto_var.get_var("norm")
    trnX, trny, tstX, tsty = auto_var.get_var("dataset")
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(np.unique(trny))
    n_channels = trnX.shape[-1]

    #trnX -= 0.5
    #tstX -= 0.5
    #img_shape = trnX.shape[1:]
    #scaler = StandardScaler(with_std=False)
    #trnX = scaler.fit_transform(trnX.reshape(len(trnX), -1)).reshape((len(trnX), ) + img_shape)
    #tstX = scaler.transform(tstX.reshape(len(tstX), -1)).reshape((len(tstX), ) + img_shape)

    result = {}
    #multigpu = True if len(trnX) > 90000 and torch.cuda.device_count() > 1 else False
    multigpu = False
    model = auto_var.get_var("model", trnX=trnX, trny=trny, multigpu=multigpu,
            n_channels=n_channels, device=device)
    model.tst_ds = (tstX, tsty)
    result['model_path'] = os.path.join('./models/experiment01', get_file_name(auto_var) + '-ep%04d.pt')
    if None:
        result['model_path'] = result['model_path'] % model.epochs
        model.load(result['model_path'])
        model.model.to(device)
    else:
        with Stopwatch("Fitting Model", logger=logger):
            history = model.fit(trnX, trny)
        mkdir_p("./models/experment01")
        model.save(result['model_path'])
        result['model_path'] = result['model_path'] % model.epochs
        result['history'] = history

    result['trn_acc'] = (model.predict(trnX) == trny).mean()
    result['tst_acc'] = (model.predict(tstX) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    attack_model = auto_var.get_var("attack", model=model, n_classes=n_classes)
    with Stopwatch("Attacking Train", logger=logger):
        adv_trnX = attack_model.perturb(trnX, trny)
    with Stopwatch("Attacking Test", logger=logger):
        adv_tstX = attack_model.perturb(tstX, tsty)
    result['adv_trn_acc'] = (model.predict(adv_trnX) == trny).mean()
    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty).mean()
    print(f"adv trn acc: {result['adv_trn_acc']}")
    print(f"adv tst acc: {result['adv_tst_acc']}")
    del attack_model

    with Stopwatch("Estimating trn Lip", logger=logger):
        trn_lip, _ = estimate_local_lip_v2(model.model, trnX, top_norm=1, btm_norm=norm,
                                    epsilon=auto_var.get_var("eps"), device=device)
    result['avg_trn_lip'] = trn_lip
    with Stopwatch("Estimating tst Lip", logger=logger):
        tst_lip, _ = estimate_local_lip_v2(model.model, tstX, top_norm=1, btm_norm=norm,
                                     epsilon=auto_var.get_var("eps"), device=device)
    result['avg_tst_lip'] = tst_lip
    print(f"avg trn lip: {result['avg_trn_lip']}")
    print(f"avg tst lip: {result['avg_tst_lip']}")

    print(result)
    return result
