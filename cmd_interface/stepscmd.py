import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import argparse
from core import steps
from cmd_interface.action.create_single_by_type_name_action import CreateSingleByTypeNameAction
from cmd_interface.action.create_list_by_type_name_action import CreateListByTypeNameAction
from core.data_store.stream_ndarray_adapter_datastore import StreamNdarrayAdapterDataStore
from core.search.exhaustive_searcher import ExhaustiveSearcher
from core.search.inverted_multi_index_searcher import InvertedMultiIndexSearcher
from core.evaluation.ground_truth import DataStoreBasedGroundTruth
from core.evaluation.retrieval_perfomance import PrecisionRecallAveragePrecisionEvaluator


def transform_step_func(args):
    steps.transform_step(args.ds_in, args.trs, args.ds_out)


def sample_step_func(args):
    steps.sampling_step(args.ds_in, args.fraction, args.ds_out)


def quantize_step_func(args):
    steps.quantize_step(args.ds_in, args.quantizer, args.ds_out)


def search_step_func(args):
    if args.search_type == "exhaustive":
        ds_ndarray_basevectors = StreamNdarrayAdapterDataStore(args.ds_basevectors,
                                                               detect_final_shape_by_first_elem=True)
        items = ds_ndarray_basevectors.get_items_sorted_by_ids()
        ids = ds_ndarray_basevectors.get_ids_sorted()
        searcher = ExhaustiveSearcher(items, ids)
    elif args.search_type == "imi":
        ds_ndarray_basevectors = StreamNdarrayAdapterDataStore(args.ds_basevectors,
                                                               detect_final_shape_by_first_elem=True)
        items = ds_ndarray_basevectors.get_items_sorted_by_ids()
        ids = ds_ndarray_basevectors.get_ids_sorted()

        ds_ndarray_centroids = StreamNdarrayAdapterDataStore(args.ds_centroids,
                                                             detect_final_shape_by_first_elem=True)
        centroids = ds_ndarray_centroids.get_items_sorted_by_ids()
        searcher = InvertedMultiIndexSearcher(ids, centroids, x=items)

    steps.search_step(args.ds_in, searcher, args.n_nearest, args.ds_out)


def evaluate_step_func(args):
    if args.groundtruth_type == "ds_based":
        ground_truth = DataStoreBasedGroundTruth(args.ds_positives, args.ds_neutrals)
        evaluator = PrecisionRecallAveragePrecisionEvaluator(ground_truth)

    steps.evaluation_step(args.ds_in, evaluator, args.ds_out)


# in_out_parser = argparse.ArgumentParser()
# in_out_parser.add_argument('--ds_in', nargs='+', action=CreateSingleByTypeNameAction)
# in_out_parser.add_argument('--ds_out', nargs='+', action=CreateSingleByTypeNameAction)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

transform_parser = subparsers.add_parser('transform')
transform_parser.add_argument('--ds_in', nargs='+', action=CreateSingleByTypeNameAction)
transform_parser.add_argument('--trs', nargs='+', action=CreateListByTypeNameAction)
transform_parser.add_argument('--ds_out', nargs='+', action=CreateSingleByTypeNameAction)
transform_parser.set_defaults(func=transform_step_func)

sample_parser = subparsers.add_parser('sample')
sample_parser.add_argument('--ds_in', nargs='+', action=CreateSingleByTypeNameAction)
sample_parser.add_argument('--fraction', type=float)
sample_parser.add_argument('--ds_out', nargs='+', action=CreateSingleByTypeNameAction)
sample_parser.set_defaults(func=sample_step_func)

quantize_parser = subparsers.add_parser('quantize')
quantize_parser.add_argument('--ds_in', nargs='+', action=CreateSingleByTypeNameAction)
quantize_parser.add_argument('--quantizer', nargs='+', action=CreateSingleByTypeNameAction)
quantize_parser.add_argument('--ds_out', nargs='+', action=CreateSingleByTypeNameAction)
quantize_parser.set_defaults(func=quantize_step_func)

search_parser = subparsers.add_parser('search')
search_parser.set_defaults(func=search_step_func)
# search_parser.add_argument('--ds_in', nargs='+', action=CreateSingleByTypeNameAction)
# search_parser.add_argument('--ds_out', nargs='+', action=CreateSingleByTypeNameAction)
# search_parser.add_argument('--n_nearest', type=int)

search_subparsers = search_parser.add_subparsers(dest="search_type")
exhaustive_search_subparser = search_subparsers.add_parser('exhaustive')
exhaustive_search_subparser.add_argument('--ds_basevectors', nargs='+', action=CreateSingleByTypeNameAction)
exhaustive_search_subparser.add_argument('--ds_in', nargs='+', action=CreateSingleByTypeNameAction)
exhaustive_search_subparser.add_argument('--ds_out', nargs='+', action=CreateSingleByTypeNameAction)
exhaustive_search_subparser.add_argument('--n_nearest', type=int)

imi_search_subparser = search_subparsers.add_parser('imi')
imi_search_subparser.add_argument('--ds_basevectors', nargs='+', action=CreateSingleByTypeNameAction)
imi_search_subparser.add_argument('--ds_centroids', nargs='+', action=CreateSingleByTypeNameAction)
imi_search_subparser.add_argument('--ds_in', nargs='+', action=CreateSingleByTypeNameAction)
imi_search_subparser.add_argument('--ds_out', nargs='+', action=CreateSingleByTypeNameAction)
imi_search_subparser.add_argument('--n_nearest', type=int)

evaluate_parser = subparsers.add_parser('evaluate')
evaluate_subparsers = evaluate_parser.add_subparsers(dest="groundtruth_type")
evaluate_parser.set_defaults(func=evaluate_step_func)

ds_based_evaluate_subparser = evaluate_subparsers.add_parser("ds_based")
ds_based_evaluate_subparser.add_argument('--ds_in', nargs='+', action=CreateSingleByTypeNameAction)
ds_based_evaluate_subparser.add_argument('--ds_out', nargs='+', action=CreateSingleByTypeNameAction)
ds_based_evaluate_subparser.add_argument('--ds_positives', nargs='+', action=CreateSingleByTypeNameAction)
ds_based_evaluate_subparser.add_argument('--ds_neutrals', nargs='*', action=CreateSingleByTypeNameAction)

plotter_parser = subparsers.add_parser('plot')
plotter_parser.add_argument('--ds_in', nargs='+', action=CreateSingleByTypeNameAction)
plotter_parser.add_argument('--trs', nargs='+', action=CreateListByTypeNameAction)
plotter_parser.add_argument('--ds_out', nargs='+', action=CreateSingleByTypeNameAction)
plotter_parser.set_defaults(func=transform_step_func)



if __name__ == '__main__':
    # cmdline = r'transform --ds_in filedirectory C:\data\images\brodatz\data.brodatz\size_213x213 --trs bytes_to_ndarray --trs ndarray_to_opencvmatrix --trs opencvmatrix_to_histogram --ds_out sqlite temp\brodatz\histogram'
    # cmdline = r'transform --ds_in filedirectory C:\data\images\brodatz\data.brodatz\size_213x213 --trs bytes_to_ndarray --trs ndarray_to_opencvmatrix --trs opencvmatrix_to_histogram --ds_out csv temp\brodatz\histogram.csv ndarray float32'
    # cmdline= r'quantize --ds_in sqlite temp\brodatz\histogram --quantizer pq 100 2 --ds_out sqlite temp\brodatz\histogram_centroids'
    # cmdline = r'search imi --n_nearest 10 --ds_in sqlite temp\brodatz\histogram --ds_basevectors sqlite temp\brodatz\histogram --ds_centroids sqlite temp\brodatz\histogram_centroids  --ds_out sqlite temp\brodatz\histogram_imi_search'
    # cmdline = r'search exhaustive --n_nearest 10 --ds_in sqlite temp\brodatz\histogram --ds_basevectors sqlite temp\brodatz\histogram  --ds_out sqlite temp\brodatz\histogram_search'
    # cmdline = r'evaluate ds_based --ds_in sqlite temp\brodatz\histogram_search --ds_positives csv temp\brodatz\brodatz_gt_positives.csv ndarray int32  --ds_out csv temp\brodatz\histogram_search_evaluation.csv'
    # cmdline = r'evaluate ds_based --ds_in sqlite temp\brodatz\histogram_search --ds_positives csv temp\brodatz\brodatz_gt_positives.csv ndarray int32  --ds_out sqlite temp\brodatz\histogram_search_evaluation.sqlite'
    # cmdline = r'transform --ds_in sqlite temp\brodatz\histogram_search_evaluation.sqlite --trs mean --ds_out sqlite temp\brodatz\histogram_search_evaluation_mean.sqlite'
    cmdline = r'transform --ds_in sqlite temp\brodatz\histogram_search_evaluation.sqlite --trs mean --ds_out csv temp\brodatz\histogram_search_evaluation_mean.csv'

    args = parser.parse_args(cmdline.split())
    # args = parser.parse_args()

    print(args)
    args.func(args)
