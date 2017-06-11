from core.transformer.items_transformer import ItemsTransformer


class FilepathToImageBytes(ItemsTransformer):
    def transform_item(self, filepath):
        with open(filepath, "rb") as binary_file:
            data = binary_file.read()
            return data
