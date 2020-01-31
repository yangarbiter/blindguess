import os

import torch
from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import set_random_seed
from lolip.utils import estimate_local_lip_v2
from lolip.variables import get_file_name


def run_record(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_state = set_random_seed(auto_var)
    norm = auto_var.get_var("norm")
    trnX, trny, tstX, tsty = auto_var.get_var("dataset")
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(np.unique(trny))
    n_channels = trnX.shape[-1]

    logger_dict = []
    counter = [0]
    idx = random_state.choice(np.arange(len(tstX)), size=100)
    ststX, ststy = tstX[idx], tsty[idx]

    def log_data(model, x, y, loss_fn):
        if counter[0] % 10 == 0:
            attack_model = auto_var.get_var("attack", model=model, n_classes=n_classes)
            tst_acc = (model.predict(ststX) == ststy).mean()
            adv_tstX = attack_model.perturb(ststX, ststy)
            adv_acc = (model.predict(adv_tstX) == ststy).mean()
            lip, _ = estimate_local_lip_v2(model.model, ststX, top_norm=1, btm_norm=norm,
                                        epsilon=auto_var.get_var("eps"), device=device)
            logger_dict.append({
                #'update_count': counter[0],
                'tst_acc': tst_acc,
                'tst_adv_acc': adv_acc,
                'tst_lip': lip,
            })

            del attack_model
        
        counter[0] += 1
        

    result = {}
    model = auto_var.get_var("model", trnX=trnX, trny=trny, multigpu=False,
                             n_channels=n_channels, trn_log_callbacks=[log_data])
    model.tst_ds = (tstX, tsty)
    with Stopwatch("Fitting Model"):
        _ = model.fit(trnX, trny)
    result['logger_dict'] = logger_dict

    result['trn_acc'] = (model.predict(trnX) == trny).mean()
    result['tst_acc'] = (model.predict(tstX) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    print(result)
    return result
