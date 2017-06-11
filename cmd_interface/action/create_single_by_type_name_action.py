import argparse
from cmd_interface.config_cmd_interface import typename_argsconverter, typename_type


def create_single_by_type_name(values):
    typename = values[0]
    args = values[1:]
    if typename in typename_argsconverter:
        args = typename_argsconverter[typename](args)

    type_ = typename_type[typename]

    # print("args", args)
    # print("*args", *args)
    obj = type_(*args)
    return obj




class CreateSingleByTypeNameAction(argparse.Action):
    def __call__(self, parser, namespace, values,
                 option_string=None):
        obj = create_single_by_type_name(values)

        setattr(namespace, self.dest, obj)
        # print()
