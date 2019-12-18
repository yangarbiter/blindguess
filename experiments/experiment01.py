from functools import partial

from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from .utils import set_random_seed


def run_experiment01(auto_var):
    random_state = set_random_seed(auto_var)
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

    attack_fn = auto_var.get_var("attack", model=model)
    with Stopwatch("Attacking"):
        adv_tstX = attack_fn(tstX)

    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty).mean()
    print(f"adversarial test acc: {result['adv_tst_acc']}")

    with Stopwatch("Estiimating Lip"):
        pass


    return result
