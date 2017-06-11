import operator
import numpy as np
from core.evaluation.ground_truth import GroundTruth


def calculate_tp_and_fp(neighbor_ids: np.ndarray, positive_ids: np.ndarray,
                        neutral_ids: np.ndarray):
    n_neighbors = len(neighbor_ids)
    tp_arr = np.empty((n_neighbors,))
    fp_arr = np.empty((n_neighbors,))

    tp = 0
    fp = 0
    for i, neighbor_id in enumerate(neighbor_ids):
        if neighbor_id in positive_ids:
            tp += 1
        elif neighbor_id not in neutral_ids:
            fp += 1
        tp_arr[i] = tp
        fp_arr[i] = fp

    return (tp_arr, fp_arr)


def calculate_threshold_retrieval_perfomances2(neighbor_ids: np.ndarray, positive_ids: np.ndarray,
                                               neutral_ids: np.ndarray, neighbors_count_cutoffs: np.ndarray = None,
                                               recall_on_cutoff=False):
    n_neighbors = len(neighbor_ids)

    if neighbors_count_cutoffs is None:
        neighbors_count_cutoffs = np.arange(1, n_neighbors + 1, dtype=np.int32)

    cutoffs_count = len(neighbors_count_cutoffs)
    precisions = np.empty((cutoffs_count,))
    recalls = np.empty((cutoffs_count,))

    tp_arr, fp_arr = calculate_tp_and_fp(neighbor_ids, positive_ids, neutral_ids)

    for i in range(cutoffs_count):
        neighbors_count_cutoff = neighbors_count_cutoffs[i]

        tp = tp_arr[neighbors_count_cutoff - 1]
        fp = fp_arr[neighbors_count_cutoff - 1]
        if tp + fp == 0:
            precisions[i] = 0
            recalls[i] = 0
        else:
            precision = tp / (tp + fp)
            if recall_on_cutoff:
                recall = tp / neighbors_count_cutoff
            else:
                recall = tp / len(positive_ids)

            precisions[i] = precision
            recalls[i] = recall

    return (neighbors_count_cutoffs, precisions, recalls)


def calculate_average_precision2(precisions: np.array, recalls: np.array):
    arr_len = len(precisions)
    ap_arr = np.empty(arr_len)
    ap = 0
    prev_recall = 0
    for i in range(arr_len):
        ap += precisions[i] * (recalls[i] - prev_recall)
        ap_arr[i] = ap
        prev_recall = recalls[i]

    return ap_arr


class RetrievalPerfomanceEvaluator:
    def calc_perfomance_results(self, id, neighbor_ids: np.ndarray):
        pass


class PrecisionRecallAveragePrecisionEvaluator(RetrievalPerfomanceEvaluator):
    def __init__(self, ground_truth: GroundTruth):
        self.ground_truth = ground_truth

    def calc_perfomance_results(self, id, neighbor_ids: np.ndarray):
        positive_ids = self.ground_truth.get_positive_ids(id)
        neutral_ids = self.ground_truth.get_neutral_ids(id)
        neighbors_count_cutoffs, precisions, recalls = calculate_threshold_retrieval_perfomances2(neighbor_ids,
                                                                                                  positive_ids,
                                                                                                  neutral_ids)
        average_precisions = calculate_average_precision2(precisions, recalls)

        return neighbors_count_cutoffs, precisions, recalls, average_precisions


def extract_perfomances_from_arr(perfomances_arr, perfomance_type):
    perfomances_arr = perfomances_arr.reshape((4, -1))
    if perfomance_type == 'precision':
        return perfomances_arr[1]
    if perfomance_type == 'recall':
        return perfomances_arr[2]
    if perfomance_type == 'mAP':
        return perfomances_arr[3]
    if perfomance_type == 'n_nearest':
        # not perfomance but ...
        return perfomances_arr[0]

# def convert_perfomances_arr_to_dict(perfomances_arr):
#     n__p_r_mAP={}
#     for n in perfomances_arr:
#         p_r_mAP=n__p_r_mAP.setdefault(n, {})
