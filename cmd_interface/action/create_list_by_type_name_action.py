import argparse
from cmd_interface.config_cmd_interface import typename_argsconverter, typename_type

class CreateListByTypeNameAction(argparse.Action):
    def __call__(self, parser, namespace, values,
                 option_string=None):
        typename = values[0]
        args = values[1:]
        if typename in typename_argsconverter:
            args = typename_argsconverter[typename](args)

        type_ = typename_type[typename]

        # print("args", args)
        # print("*args", *args)
        obj = type_(*args)

        curlist = getattr(namespace, self.dest)
        if curlist is None:
            curlist = []
        curlist.append(obj)

        setattr(namespace, self.dest, curlist)
        # print()