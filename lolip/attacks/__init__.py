from functools import partial

import numpy as np
from autovar.base import RegisteringChoiceType, VariableClass, register_var

class AttackVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines which attack method to use."""
    var_name = "attack"

    @register_var()
    @staticmethod
    def pgd(auto_var, inter_var, model):
        from .torch.projected_gradient_descent import ProjectedGradientDescent
        nb_iter=10
        return ProjectedGradientDescent(
            model_fn=model.model,
            norm=auto_var.get_var("norm"),
            #lbl_enc=inter_var['lbl_enc'],
            eps=auto_var.get_var("eps"),
            eps_iter=auto_var.get_var("eps")*2/nb_iter,
            nb_iter=nb_iter,
        )