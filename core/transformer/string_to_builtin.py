from core.transformer.items_transformer import ItemsTransformer


class StringToBuiltin(ItemsTransformer):
    def __init__(self, typestr):
        self.type_ = eval(typestr)

    def transform_item(self, item):
        transformed = self.type_(item)
        return transformed
