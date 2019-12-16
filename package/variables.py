import logging
from functools import partial

from autovar import AutoVar
from autovar.hooks import check_result_file_exist, save_result_to_file
from autovar.hooks import default_get_file_name as get_file_name

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
