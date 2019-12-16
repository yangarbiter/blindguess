
from autovar.base import RegisteringChoiceType, VariableClass, register_var

class ExampleVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Example Variable Class"""
    var_name = "example"

    @register_var()
    @staticmethod
    def exp(auto_var):
        pass
