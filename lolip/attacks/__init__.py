from functools import partial

import numpy as np
from autovar.base import RegisteringChoiceType, VariableClass, register_var

class AttackVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines which attack method to use."""
    var_name = "attack"

    @register_var()
    @staticmethod
    def pgd(auto_var, inter_var, model, n_classes, clip_min=None, clip_max=None):
        from .torch.projected_gradient_descent import ProjectedGradientDescent
        nb_iter=10
        return ProjectedGradientDescent(
            model_fn=model.model,
            norm=auto_var.get_var("norm"),
            clip_min=clip_min,
            clip_max=clip_max,
            #lbl_enc=inter_var['lbl_enc'],
            eps=auto_var.get_var("eps"),
            eps_iter=auto_var.get_var("eps")*2/nb_iter,
            nb_iter=nb_iter,
        )

    @register_var()
    @staticmethod
    def multitarget(auto_var, inter_var, model, n_classes):
        from .torch.multi_target import MultiTarget
        nb_iter=10
        return MultiTarget(
            n_classes=n_classes,
            model_fn=model.model,
            norm=auto_var.get_var("norm"),
            eps=auto_var.get_var("eps"),
            eps_iter=auto_var.get_var("eps")*2/nb_iter,
            nb_iter=nb_iter,
        )

    @register_var()
    @staticmethod
    def hongmt(auto_var, model, n_classes):
        from .torch.hong_mt import HongMultiTarget
        nb_iter=20
        return HongMultiTarget(
            n_classes=n_classes,
            model_fn=model.model,
            norm=auto_var.get_var("norm"),
            eps=auto_var.get_var("eps"),
            eps_iter=auto_var.get_var("eps")*2/nb_iter,
            nb_iter=nb_iter,
        )

    @register_var()
    @staticmethod
    def hongpgd(auto_var, model, n_classes):
        from .torch.hong_pgd import HongPGD
        nb_iter=20
        return HongPGD(
            n_classes=n_classes,
            model_fn=model.model,
            norm=auto_var.get_var("norm"),
            eps=auto_var.get_var("eps"),
            eps_iter=auto_var.get_var("eps")*2/nb_iter,
            nb_iter=nb_iter,
        )
