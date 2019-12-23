import os

from autovar.base import RegisteringChoiceType, VariableClass, register_var

DEBUG = int(os.getenv("DEBUG", 0))

def get_hyper(name, loss, arch, dataset_name):
    ret = {}
    if 'CNN' in arch and ('mnist' in dataset_name or 'fashion' in dataset_name):
        ret['epochs'] = 200
        ret['learning_rate'] = 1e-4
        ret['momentum'] = 0.9
        ret['batch_size'] = 64
    elif 'ResNet' in arch or 'WRN' in arch:
        ret['epochs'] = 200
        if 'advce' in loss:
            ret['learning_rate'] = 1e-3
        else:
            ret['learning_rate'] = 1e-2
        ret['batch_size'] = 64
    else:
        ret['epochs'] = 500
        if 'trades' in name:
            ret['learning_rate'] = 1e-2
        elif 'sgd' in name:
            ret['learning_rate'] = 5e-1
        else:
            ret['learning_rate'] = 1e-1
        ret['batch_size'] = 128

    if DEBUG:
        ret['epochs'] = 2
    return ret

class ModelVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Model Variable Class"""
    var_name = "model"

    @register_var(argument=r'(?P<train>[a-zA-Z0-9]+-)?(?P<loss>[a-zA-Z0-9]+)-tor-(?P<arch>[a-zA-Z0-9_]+)(?P<hyper>-[a-zA-Z0-9]+)?')
    @staticmethod
    def torch_model(auto_var, inter_var, train, loss, arch, hyper, trnX, trny):
        from .torch_model import TorchModel

        train = train[:-1] if train else None

        n_features = trnX.shape[1:]
        n_classes = len(set(trny))
        dataset_name = auto_var.get_variable_name('dataset')

        params: dict = get_hyper(hyper, loss, arch, dataset_name)
        params['eps'] = auto_var.get_var("eps")
        params['norm'] = auto_var.get_var("norm")
        params['loss_name'] = loss
        params['n_features'] = n_features
        params['n_classes'] = n_classes
        params['train_type'] = train
        params['architecture'] = arch
        #params['ckpt_dir'] = f"./checkpoints/{auto_var.get_variable_name('model')}"
        #if hyper == "gd":
        #    params['batch_size'] = len(trnX)

        model = TorchModel(
            lbl_enc=inter_var['lbl_enc'],
            **params,
        )
        return model
