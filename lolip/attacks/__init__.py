from functools import partial

import numpy as np
from autovar.base import RegisteringChoiceType, VariableClass, register_var

class AttackVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines which attack method to use."""
    var_name = "attack"

    @register_var()
    @staticmethod
    def fftspatial(auto_var, model, n_classes, clip_min=None, clip_max=None, device=None):
        from .torch.spatials import FFTAttackModel
        return FFTAttackModel(
            model_fn=model.model,
            perturb_iters=10,
            step_size=0.05,
            rot_constraint=0,
            trans_constraint=0.2,
            scale_constraint=0,
            batch_size=128,
        )

    @register_var()
    @staticmethod
    def fospatial(auto_var, model, n_classes, clip_min=None, clip_max=None, device=None):
        from .torch.spatials import FirstOrderAttackModel
        return FirstOrderAttackModel(
            model_fn=model.model,
            perturb_iters=10,
            step_size=0.05,
            rot_constraint=0,
            trans_constraint=0.2,
            scale_constraint=0,
            batch_size=128,
        )

    @register_var()
    @staticmethod
    def gridspatial(auto_var, model, n_classes, clip_min=None, clip_max=None, device=None):
        from .torch.spatials import GridAttackModel
        return GridAttackModel(
            model_fn=model.model,
            rot_constraint=0,
            trans_constraint=0.2,
            scale_constraint=0,
            batch_size=128,
        )

    @register_var()
    @staticmethod
    def pgd(auto_var, inter_var, model, n_classes, clip_min=None, clip_max=None, device=None):
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
    def multitarget(auto_var, model, n_classes, clip_min=None, clip_max=None, device=None):
        from .torch.multi_target import MultiTarget
        nb_iter=20
        return MultiTarget(
            n_classes=n_classes,
            model_fn=model.model,
            norm=auto_var.get_var("norm"),
            clip_min=clip_min,
            clip_max=clip_max,
            eps=auto_var.get_var("eps"),
            eps_iter=auto_var.get_var("eps")*2/nb_iter,
            nb_iter=nb_iter,
            batch_size=128,
        )

    @register_var()
    @staticmethod
    def mtv2(auto_var, model, n_classes, clip_min=None, clip_max=None):
        from .torch.mtv2 import MultiTargetV2
        nb_iter=20
        return MultiTargetV2(
            n_classes=n_classes,
            model_fn=model.model,
            norm=auto_var.get_var("norm"),
            eps=auto_var.get_var("eps"),
            eps_iter=auto_var.get_var("eps")*2/nb_iter,
            nb_iter=nb_iter,
            clip_min=clip_min,
            clip_max=clip_max,
            batch_size=256,
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
