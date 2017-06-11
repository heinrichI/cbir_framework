from contextlib import ExitStack

from core.data_store.datastore import DataStore


def print_ds_items_info(ds: DataStore, first_items_to_print=3, print_shape_only=True):
    with ExitStack() as stack:
        if hasattr(ds, '__enter__'):
            stack.enter_context(ds)
        items_count = ds.get_count()
        print("count of items in ds: ", items_count)
        items = ds.get_items_sorted_by_ids()
        items = iter(items)
        if items_count < first_items_to_print:
            first_items_to_print = items_count
        for i in range(first_items_to_print):
            item = next(items)
            if hasattr(item, 'shape'):
                item_info = item.shape
                print("shape of item[{0}]: ".format(i), item_info)
            if not print_shape_only:
                item_info = item
                print("item[{0}]: ".format(i), item_info)


def print_ds_items_info_dispatch(ds: DataStore, print_ds_out_info=None):
    if print_ds_out_info == 'shape':
        print_ds_items_info(ds, print_shape_only=True)
    elif print_ds_out_info == 'ndarray':
        print_ds_items_info(ds, print_shape_only=False)
