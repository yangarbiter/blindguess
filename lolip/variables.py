import logging
from functools import partial

from autovar import AutoVar
from autovar.base import RegisteringChoiceType, register_var, VariableClass
from autovar.hooks import check_result_file_exist, save_result_to_file
from autovar.hooks import default_get_file_name as get_file_name

from .datasets import DatasetVarClass

auto_var = AutoVar(
    logging_level=logging.INFO,
    before_experiment_hooks=[partial(check_result_file_exist, get_name_fn=get_file_name)],
    after_experiment_hooks=[partial(save_result_to_file, get_name_fn=get_file_name)],
    settings={
        'file_format': 'pickle',
        'server_url': '',
        'result_file_dir': './results/'
    }
)

class NormVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines which distance measure to use for attack."""
    var_name = "norm"

    @register_var()
    @staticmethod
    def inf(auto_var):
        """L infinity norm"""
        return np.inf

    @register_var(argument='2')
    @staticmethod
    def l2(auto_var):
        """L2 norm"""
        return 2

    @register_var(argument='1')
    @staticmethod
    def l1(auto_var):
        """L1 norm"""
        return 1

auto_var.add_variable_class(NormVarClass())
auto_var.add_variable_class(DatasetVarClass())
auto_var.add_variable('random_seed', int)
auto_var.add_variable('eps', float)

#from autovar.base import RegisteringChoiceType, VariableClass, register_var
#class ExampleVarClass(VariableClass, metaclass=RegisteringChoiceType):
#    """Example Variable Class"""
#    var_name = "example"
#
#    @register_var()
#    @staticmethod
#    def exp(auto_var):
#        pass

#auto_var.add_variable_class(ExampleVarClass())
