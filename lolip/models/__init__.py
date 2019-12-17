
from autovar.base import RegisteringChoiceType, VariableClass, register_var

class ModelVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Model Variable Class"""
    var_name = "model"

    @register_var()
    @staticmethod
    def trades(auto_var):
        pass
