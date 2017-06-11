from core.transformer.items_transformer import ItemsTransformer

class ParametrizedItemsTransformer(ItemsTransformer):
    def __init__(self, item_transform_func, *args, **kwargs):
        self.item_transform_func = item_transform_func
        self.args = args
        self.kwargs = kwargs

    def transform_item(self, item):
        return self.item_transform_func(item, *self.args, **self.kwargs)

    def getParamsInfo(self):
        pass
