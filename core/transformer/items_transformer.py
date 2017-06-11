import collections



class ItemsTransformer():
    def transform(self, items: collections.Iterable):
        return map(self.transform_item, items)

    def transform_item(self):
        pass

    def get_result_item_info(self):
        pass


