from contextlib import ExitStack

import numpy as np
from core.data_store.datastore import DataStore
from core.data_store.stream_ndarray_adapter_datastore import StreamNdarrayAdapterDataStore
from core.search.searcher import Searcher

from core.common.itertools_utils import pipe_line
from core.quantization.quantizer import Quantizer
from core.evaluation.retrieval_perfomance import RetrievalPerfomanceEvaluator
from core.transformer.mean_calculator import MeanCalculator
from core.common.ds_utils import print_ds_items_info_dispatch
from core.common.data_wrap import DataWrap
from core.common.file_utils import make_if_not_exists
import itertools


def transform_step(data_store_in: DataStore, transformers, data_store_out: DataStore, force=False,
                   print_ds_out_info=None) -> None:
    """

          data_store_in and data_store_out must be of same type: both ndarray or both stream.
          transformers must be of ds type
    """
    if not force and data_store_out.already_exists():
        return
    # TODO add dataframe abstraction and make any step work with any type of data(stream and ndarray)
    with ExitStack() as stack:
        if hasattr(data_store_in, '__enter__'):
            stack.enter_context(data_store_in)
        if hasattr(data_store_out, '__enter__'):
            stack.enter_context(data_store_out)

        count_ = data_store_in.get_count()
        items_in_sorted_by_ids = data_store_in.get_items_sorted_by_ids()

        items_out_sorted_by_ids = pipe_line(items_in_sorted_by_ids, transformers, count_)

        # ids_sorted = data_store_in.get_ids_sorted()
        # print(ids_sorted)
        # data_store_out.save_items_sorted_by_ids(items_out_sorted_by_ids, ids_sorted)
        data_store_out.save_items_sorted_by_ids(items_out_sorted_by_ids)

    print_ds_items_info_dispatch(data_store_out, print_ds_out_info)


def sampling_step(data_store_in: DataStore, sample_part, data_store_out: DataStore, force=False,
                  print_ds_out_info=None) -> None:
    """
        data_store_in - stream or ndarray ds
        data_store_out - stream or ndarray ds
    """
    if not force and data_store_out.already_exists():
        return
    with ExitStack() as stack:
        if hasattr(data_store_in, '__enter__'):
            stack.enter_context(data_store_in)
        if hasattr(data_store_out, '__enter__'):
            stack.enter_context(data_store_out)
        count_ = data_store_in.get_count()
        sample_size = int(count_ * sample_part)

        ds_ndarray_in = StreamNdarrayAdapterDataStore(data_store_in, detect_final_shape_by_first_elem=True)
        ids_sorted_ndarray = ds_ndarray_in.get_ids_sorted()
        id_ndarray_sample = np.random.choice(ids_sorted_ndarray, sample_size, replace=False)
        id_ndarray_sample.sort()
        id_ndarray_sample_sorted = id_ndarray_sample

        sample_items_sorted_by_ids = ds_ndarray_in.get_items_sorted_by_ids(id_ndarray_sample_sorted)

        ds_ndarray_out = StreamNdarrayAdapterDataStore(data_store_out, detect_final_shape_by_first_elem=True)
        ds_ndarray_out.save_items_sorted_by_ids(sample_items_sorted_by_ids)

    print_ds_items_info_dispatch(data_store_out, print_ds_out_info)


def quantize_step(data_store_in: DataStore, quantizer: Quantizer, data_store_out: DataStore, force=False,
                  print_ds_out_info=None, quantization_info_ds: DataStore = None) -> None:
    """
        data_store_in - stream or ndarray ds
        data_store_out - stream or ndarray ds
    """
    if not force and data_store_out.already_exists():
        return

    ds_ndarray_in = StreamNdarrayAdapterDataStore(data_store_in, detect_final_shape_by_first_elem=True)
    items_ndarray = ds_ndarray_in.get_items_sorted_by_ids()
    quantizer.fit(items_ndarray)
    cluster_centers_ndarray = quantizer.get_cluster_centers()

    ds_ndarray_out = StreamNdarrayAdapterDataStore(data_store_out, detect_final_shape_by_first_elem=True)
    ds_ndarray_out.save_items_sorted_by_ids(cluster_centers_ndarray)

    if quantization_info_ds is not None:
        with ExitStack() as stack:
            if hasattr(quantization_info_ds, '__enter__'):
                stack.enter_context(quantization_info_ds)
            items = [quantizer.get_quantization_info()]
            quantization_info_ds.save_items_sorted_by_ids(items)

    print_ds_items_info_dispatch(data_store_out, print_ds_out_info)


def search_step(data_store_in: DataStore, searcher_: Searcher, n_nearest, data_store_out: DataStore,
                force=False, print_ds_out_info=None) -> None:
    """
        data_store_in - stream ds
        data_store_out - stream or ndarray ds
    """
    if not force and data_store_out.already_exists():
        return

    with ExitStack() as stack:
        if hasattr(data_store_in, '__enter__'):
            stack.enter_context(data_store_in)
        items = data_store_in.get_items_sorted_by_ids()
        items_count = data_store_in.get_count()

        dw = DataWrap(items, items_count=items_count)
        dw.chunkify(chunk_size=1000)
        out_stream = map(searcher_.find_nearest_ids, dw.data_stream, itertools.repeat(n_nearest))
        out_dw = DataWrap(out_stream, is_stream_chunkified=True)
        out_dw.dechunkify()

        with ExitStack() as stack:
            if hasattr(data_store_out, '__enter__'):
                stack.enter_context(data_store_out)
            data_store_out.save_items_sorted_by_ids(out_dw.data_stream)

    print_ds_items_info_dispatch(data_store_out, print_ds_out_info)


def evaluation_step(data_store_in: DataStore, evaluator: RetrievalPerfomanceEvaluator,
                    data_store_out: DataStore, force=False, mean=True, print_ds_out_info=None) -> None:
    """
        data_store_in - stream or ndarray ds
        data_store_out - stream or ndarray ds
    """
    if not force and data_store_out.already_exists():
        return
    with ExitStack() as stack:
        if hasattr(data_store_in, '__enter__'):
            stack.enter_context(data_store_in)
        if hasattr(data_store_out, '__enter__'):
            stack.enter_context(data_store_out)

        ids_sorted_stream = data_store_in.get_ids_sorted()
        neighbor_ids_stream = data_store_in.get_items_sorted_by_ids()
        perfomance_results_stream = map(evaluator.calc_perfomance_results, ids_sorted_stream, neighbor_ids_stream)
        perfomance_results_stream = map(lambda perfomance_results_tuple: np.array(perfomance_results_tuple),
                                        perfomance_results_stream)

        ids_sorted_stream_ = data_store_in.get_ids_sorted()
        if not mean:
            data_store_out.save_items_sorted_by_ids(perfomance_results_stream, ids_sorted_stream_)
        else:
            mean_perfomance_results_stream = MeanCalculator().transform(perfomance_results_stream)
            data_store_out.save_items_sorted_by_ids(mean_perfomance_results_stream, ids_sorted_stream_)

    print_ds_items_info_dispatch(data_store_out, print_ds_out_info)


def plotting_step(results_ds_arr, results_x_pos, results_y_pos, labels, xlabel, ylabel, legend_loc=None,
                  save_to_file=None):
    results_ndarray_ds_arr = [StreamNdarrayAdapterDataStore(results_ds, detect_final_shape_by_first_elem=True) for
                              results_ds in
                              results_ds_arr]
    items0 = results_ndarray_ds_arr[0].get_items_sorted_by_ids()
    results0 = items0[0]
    x = results0[results_x_pos]
    x = x.ravel()
    # print("x_shape", x.shape)

    Y = [results_ndarray_ds.get_items_sorted_by_ids()[0][results_y_pos].reshape((-1, 1)) for results_ndarray_ds in
         results_ndarray_ds_arr]

    import matplotlib.pyplot as plt
    plt.close()
    dpi = 100
    plt.figure(figsize=(25, 15))
    label_iter = iter(labels)
    for y in Y:
        y = y.ravel()
        # print("y_shape", y.shape)
        if len(x) > 30:
            plt.semilogx(x, y, linewidth=2.0, label=next(label_iter), basex=2)
            # old_ticks = plt.xticks()[0]
            # last_tick = np.array([len(x)])
            # new_ticks = np.concatenate((old_ticks, last_tick))
            # plt.xticks(new_ticks)
        else:
            plt.plot(x, y, linewidth=2.0, label=next(label_iter))
            plt.xticks(x)

    plt.legend(loc=legend_loc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(np.linspace(0, 1, 21))

    plt.grid(True)

    if save_to_file is None:
        plt.show()
    else:
        make_if_not_exists(save_to_file)
        plt.savefig(save_to_file)


def plotting_step2(label__x__y: dict, label_prefix, xlabel, ylabel, title='', legend_loc=None,
                   save_to_file=None):
    label_x_y = []
    logscale = False
    for label in sorted(label__x__y):
        x__y = label__x__y[label]
        x = sorted(list(x__y))
        if max(x) / min(x) > 30:
            logscale = True
        y = []
        for x_ in x:
            y.append(x__y[x_])
        label_x_y.append((label_prefix + str(label), x, y))

    import matplotlib.pyplot as plt
    plt.close()
    dpi = 100
    plt.figure(figsize=(14, 7))
    # from matplotlib.markers import MarkerStyle
    # mar = MarkerStyle.markers
    filled_markers = itertools.cycle(('o', 'v', 's', 'p', '*', '<', 'h', '>', 'H', 'D', 'd', 'P', '^', 'X'))
    for label_x_y_ in label_x_y:
        label = label_x_y_[0]
        x = label_x_y_[1]
        y = label_x_y_[2]
        if logscale:
            plt.semilogx(x, y, linewidth=2.0, label=label, basex=2, marker=next(filled_markers), markersize=5.0)
        else:
            plt.plot(x, y, linewidth=2.0, label=label, marker=next(filled_markers), markersize=10.0)

    if logscale:
        pass
    # old_ticks = plt.xticks()[0]
    # last_tick = np.array([len(x)])
    # new_ticks = np.concatenate((old_ticks, last_tick))
    # plt.xticks(new_ticks)
    else:
        plt.xticks(x)

    lgd = plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(np.linspace(0, 1, 21))

    plt.grid(True)

    # save_to_file = None
    if save_to_file is None:
        plt.show()
    else:
        make_if_not_exists(save_to_file)
        plt.savefig(save_to_file, bbox_extra_artists=(lgd,), bbox_inches='tight')


def plotting_step3(subplotvalue__label__x__y: dict, subplotvalue__prefix, label_callback, xlabel, ylabel, title='',
                   legend_loc=None,
                   save_to_file=None, label__kwargs=None, bar=False):
    import matplotlib.pyplot as plt
    plt.close()
    dpi = 120
    fig = plt.figure(figsize=(20, 15))
    # fig=plt.figure()
    fig.suptitle(title)
    cols = len(subplotvalue__label__x__y.keys()) ** 0.5
    for i, subplotvalue in enumerate(sorted(subplotvalue__label__x__y.keys())):
        subplot = plt.subplot(cols, cols, i + 1)
        subplotname = subplotvalue__prefix + str(subplotvalue)
        plot_(plt, subplotvalue__label__x__y[subplotvalue], label_callback, xlabel, ylabel, subplotname, legend_loc,
              label__kwargs,bar)

    plt.tight_layout()
    # save_to_file = None
    if save_to_file is None:
        plt.show()
    else:
        make_if_not_exists(save_to_file)
        # plt.savefig(save_to_file, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig(save_to_file)


def plot_(subplot, label__x__y: dict, label_callback, xlabel, ylabel, title='', legend_loc=None, label__kwargs=None,
          bar=False):
    label_x_y = []
    logscale = False
    x_set = set()
    for label in sorted(label__x__y):
        x__y = label__x__y[label]
        x = sorted(list(x__y))
        x_set.update(x)
        # print(max(x_set), min(x_set))
        try:
            if max(x_set) / min(x_set) > 30:
                logscale = True
        except:
            pass
        y = []
        for x_ in x:
            y.append(x__y[x_])
        label_x_y.append((label, x, y))

    filled_markers = itertools.cycle(('o', 'v', 's', 'p', '*', '<', 'h', '>', 'H', 'D', 'd', 'P', '^', 'X'))
    for i, label_x_y_ in enumerate(label_x_y, 0):
        label = label_x_y_[0]
        x = label_x_y_[1]
        y = label_x_y_[2]
        if label__kwargs is not None:
            kwargs = label__kwargs.get(label, {})
        else:
            kwargs = {}

        if isinstance(x[0], str):
            x_range = range(len(x))
            subplot.xticks(x_range, x)
            x = np.arange(len(x))

        if bar:
            width = 0.09
            subplot.xticks(x - width / 2 + width * len(label_x_y) / 2)
            subplot.bar(x + width * i, y, width, label=label_callback(label), align='center')
        else:
            if logscale:
                subplot.semilogx(x, y, label=label_callback(label), basex=2, marker=next(filled_markers), **kwargs)
            else:
                subplot.plot(x, y, label=label_callback(label), marker=next(filled_markers), **kwargs)

    try:
        if logscale:
            subplot.xticks(sorted(x_set))
        else:
            subplot.xticks(sorted(x_set))
    except:
        pass

    # lgd = subplot.legend(loc=2, bbox_to_anchor=(1.05, 1))
    lgd = subplot.legend()
    subplot.title(title)
    subplot.xlabel(xlabel)
    subplot.ylabel(ylabel)
    subplot.yticks(np.linspace(0, 1, 41))

    subplot.grid(True)
    subplot.tight_layout()
