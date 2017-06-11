from core.transformer.items_transformer import ItemsTransformer


class ArrayToReshapedArray(ItemsTransformer):
    def __init__(self, result_shape):
        self.result_shape = result_shape

    def transform_item(self, item):
        return item.reshape(self.result_shape)


