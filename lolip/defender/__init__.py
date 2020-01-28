from functools import partial

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from autovar.base import RegisteringChoiceType, VariableClass, register_var


class DefenderVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """"""
    var_name = "defender"

    #@register_var(argument=r'approxAP-(?P<eps>\d+)')
    #@staticmethod
    #def approx_adversarial_pruning(auto_var, eps, norm):
    #    from .defense import approx_ap
    #    return partial(approx_ap,
    #        eps=int(eps)*0.01,
    #        sep_measure=auto_var.get_var_with_argument("ord", ord)
    #    )

    @register_var(argument=r'AP-(?P<eps>[0-9]*\.[0-9]+|[0-9]+)-(?P<ord>[a-zA-Z0-9]+)')
    @staticmethod
    def adversarial_pruning(auto_var, eps):
        from .defense import adversarial_pruning
        eps = float(eps)
        norm = auto_var.get_var("norm")
        return partial(adversarial_pruning,
            eps=eps,
            sep_measure=norm,
        )

    @register_var(argument=r'AP-(?P<eps>\d+)-(?P<ord>[a-zA-Z0-9]+)')
    @staticmethod
    def soft_ap(auto_var, eps, ord):
        from .defense import adversarial_pruning
        eps = float(eps)
        norm = auto_var.get_var("norm")
        return partial(adversarial_pruning,
            eps=eps,
            sep_measure=norm,
        )

    #@register_var()
    #@staticmethod
    #def identity(auto_var):
    #    return lambda a, b: (a, b)

    #@register_var(argument=r'AT-(?P<eps>\d+)-(?P<ord>[a-zA-Z0-9]+)-(?P<n_iters>\d+)-(?P<model_name>[a-zA-Z0-9_]+)')
    #@staticmethod
    #def adversarial_training(auto_var, eps, ord, n_iters, model_name):
    #    from .defense import iter_linear_opt_at
    #    def get_model_fn(**kwargs):
    #        return auto_var.get_var_with_argument("model", model_name, **kwargs)
    #    #attack = auto_var.get_var_with_argument("attack", attack_name)
    #    n_iters = int(n_iters)
    #    return partial(iter_linear_opt_at,
    #        eps=int(eps)*0.01,
    #        get_model_fn=get_model_fn,
    #        n_iters=n_iters,
    #        #attack_model=attack,
    #        model_name=model_name,
    #        norm=auto_var.get_var_with_argument("ord", ord),
    #    )
