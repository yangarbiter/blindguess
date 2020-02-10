import os

from autovar.base import RegisteringChoiceType, VariableClass, register_var
import numpy as np


DEBUG = int(os.getenv("DEBUG", 0))

class ModelVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Model Variable Class"""
    var_name = "model"

    @register_var(argument=r'(?P<dataaug>[a-zA-Z0-9]+-)?(?P<loss>[a-zA-Z0-9]+)-tor2-(?P<arch>[a-zA-Z0-9_]+)(?P<hyper>-[a-zA-Z0-9\.]+)?')
    @staticmethod
    def torch_model_v2(auto_var, inter_var, dataaug, loss, arch, hyper, trnX, trny, n_channels, device, multigpu=False):
        from .torch_model_v2 import TorchModelV2

        dataaug = dataaug[:-1] if dataaug else None

        n_features = trnX.shape[1:]
        n_classes = len(np.unique(trny))
        dataset_name = auto_var.get_variable_name('dataset')

        params = {}
        params['eps'] = auto_var.get_var("eps")
        params['norm'] = auto_var.get_var("norm")
        params['loss_name'] = loss
        params['n_features'] = n_features
        params['n_classes'] = n_classes
        params['architecture'] = arch
        params['multigpu'] = multigpu
        params['n_channels'] = n_channels
        params['dataaug'] = dataaug

        params['learning_rate'] = auto_var.get_var("learning_rate")
        params['epochs'] = auto_var.get_var("epochs")
        params['momentum'] = auto_var.get_var("momentum")
        params['optimizer'] = auto_var.get_var("optimizer")
        params['batch_size'] = auto_var.get_var("batch_size")
        params['weight_decay'] = auto_var.get_var("weight_decay")

        #params['ckpt_dir'] = f"./checkpoints/{auto_var.get_variable_name('model')}"
        #if hyper == "gd":
        #    params['batch_size'] = len(trnX)

        model = TorchModelV2(
            lbl_enc=inter_var['lbl_enc'],
            **params,
        )
        return model
