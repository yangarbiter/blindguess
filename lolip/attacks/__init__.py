from functools import partial

import numpy as np
from autovar.base import RegisteringChoiceType, VariableClass, register_var

class AttackVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines which attack method to use."""
    var_name = "attack"

    @register_var()
    @staticmethod
    def pgd(auto_var, inter_var, model, **kwargs):
        from .torch.projected_gradient_descent import ProjectedGradientDescent
        return partial(ProjectedGradientDescent,
            model_fn=model.model,
            norm=auto_var.get_var("norm"),
            lbl_enc=inter_var['lbl_enc'],
            **kwargs
        )