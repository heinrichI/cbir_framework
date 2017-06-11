from core.transformer.items_transformer import ItemsTransformer

class BuiltinToString(ItemsTransformer):
    def transform_item(self, item):
        return str(item)