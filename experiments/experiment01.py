import os

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

def run_experiment01(auto_var):
    _ = set_random_seed(auto_var)
    norm = auto_var.get_var("norm")
    trnX, trny, tstX, tsty = auto_var.get_var("dataset")
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(np.unique(trny))

    #trnX -= 0.5
    #tstX -= 0.5
    #img_shape = trnX.shape[1:]
    #scaler = StandardScaler(with_std=False)
    #trnX = scaler.fit_transform(trnX.reshape(len(trnX), -1)).reshape(
    #        (len(trnX), ) + img_shape)
    #tstX = scaler.transform(tstX.reshape(len(tstX), -1)).reshape((len(tstX), ) + img_shape)

    result = {}
    model = auto_var.get_var("model", trnX=trnX, trny=trny)
    model.tst_ds = (tstX, tsty)
    result['model_path'] = os.path.join('./models', get_file_name(auto_var) + '.pt')
    if None:
        model.load(result['model_path'])
        model.model.cuda()
    else:
        with Stopwatch("Fitting Model"):
            history = model.fit(trnX, trny)
        model.save(result['model_path'])
        result['history'] = history

    result['trn_acc'] = (model.predict(trnX) == trny).mean()
    result['tst_acc'] = (model.predict(tstX) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    attack_model = auto_var.get_var("attack", model=model, n_classes=n_classes)
    with Stopwatch("Attacking"):
        adv_trnX = attack_model.perturb(trnX)
        adv_tstX = attack_model.perturb(tstX)
    result['adv_trn_acc'] = (model.predict(adv_trnX) == trny).mean()
    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty).mean()
    print(f"adv trn acc: {result['adv_trn_acc']}")
    print(f"adv tst acc: {result['adv_tst_acc']}")
    del attack_model

    with Stopwatch("Estimating trn Lip"):
        trn_lip = estimate_local_lip_v2(model.model, trnX, top_norm=2, btm_norm=norm,
                                     epsilon=auto_var.get_var("eps"))
    result['avg_trn_lip'] = calc_lip(model, trnX, trn_lip, top_norm=2, btm_norm=norm).mean()
    with Stopwatch("Estimating tst Lip"):
        tst_lip = estimate_local_lip_v2(model.model, tstX, top_norm=2, btm_norm=norm,
                                     epsilon=auto_var.get_var("eps"))
    result['avg_tst_lip'] = calc_lip(model, tstX, tst_lip, top_norm=2, btm_norm=norm).mean()
    print(f"avg trn lip: {result['avg_trn_lip']}")
    print(f"avg tst lip: {result['avg_tst_lip']}")

    print(result)
    return result
