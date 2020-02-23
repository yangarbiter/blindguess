import os
import logging

import torch
from bistiming import Stopwatch
from mkdir_p import mkdir_p
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .utils import set_random_seed, load_model
from lolip.utils import estimate_local_lip_v2
from lolip.variables import get_file_name

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def run_spatial(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = set_random_seed(auto_var)
    #norm = auto_var.get_var("norm")
    trnX, trny, tstX, tsty = auto_var.get_var("dataset")
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(np.unique(trny))
    n_channels = trnX.shape[-1]

    result = {}
    #multigpu = True if len(trnX) > 90000 and torch.cuda.device_count() > 1 else False
    multigpu = False
    try:
        model_path, model = load_model(
            auto_var, trnX, trny, tstX, tsty, n_channels, model_dir="./models/experiment01", device=device)
        model.model.to(device)
        result['model_path'] = model_path
    except:
        del model
        logger.info("Model not trained yet, retrain the model")
        mkdir_p("./models/experiment01")
        result['model_path'] = os.path.join(
            './models/experiment01', get_file_name(auto_var) + '-ep%04d.pt')
        result['model_path'] = result['model_path'].replace(
            auto_var.get_variable_name("attack"), "pgd")

        model = auto_var.get_var("model", trnX=trnX, trny=trny, multigpu=multigpu,
                n_channels=n_channels, device=device)
        model.tst_ds = (tstX, tsty)
        with Stopwatch("Fitting Model", logger=logger):
            history = model.fit(trnX, trny)
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

    print(result)
    return result
