import numpy as np
from cmd_interface.converter.args_converter import ArgsConverter

a = np.empty(1)

class EvalArgsConverter(ArgsConverter):
    def __call__(self, args_list):
        evaluated_args_list = [eval(arg) for arg in args_list]
        return evaluated_args_list
