from cmd_interface.converter.args_converter import ArgsConverter


class PQQuantizerArgsConverter(ArgsConverter):
    def __call__(self, args_list):
        converters = [int, int, int, str, int, int]
        converted_args_list = [converter(arg) for converter, arg in zip(converters, args_list)]
        return converted_args_list
