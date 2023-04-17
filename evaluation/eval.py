from collections import OrderedDict
import numpy as np


DISTANCE_CUTOFF = 25
ALL_CLS_IDX = (1, 2)


def _check_validity(pred_all, gt_all):
    """ Check validity of arguments (submission output and grount-truth).

    Parameters
    ----------
    pred_all: List[List[Tuple(int, int, int, float)]]
        List of predictions, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each predicted cell consist of x, y, cls, prob.
    gt_all: List[List[Tuple(int, int, int)]]
        List of GTs, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each GT cell consist of x, y, cls.
    """

    # Number of samples
    assert len(pred_all) == len(gt_all)

    for pred, gt in zip(pred_all, gt_all):
        # x should be 0 ~ 1023
        assert all([0 <= point[0] <= 1023 and type(point[0]) == int for point in gt])
        assert all([0 <= point[1] <= 1023 and type(point[1]) == int for point in gt])

        # y should be 0 ~ 1023
        assert all([0 <= point[0] <= 1023 and type(point[0]) == int for point in pred])
        assert all([0 <= point[1] <= 1023 and type(point[1]) == int for point in pred])

        # class should be 1 or 2
        assert all([point[2] in ALL_CLS_IDX for point in gt])
        assert all([point[2] in ALL_CLS_IDX for point in pred])

        # probability should be 0.0 ~ 1.0
        assert all([0.0 <= point[3] <= 1.0 for point in pred])


def _preprocess_distance_and_confidence(pred_all, gt_all):
    """ Preprocess distance and confidence used for F1 calculation.

    Parameters
    ----------
    pred_all: List[List[Tuple(int, int, int, float)]]
        List of predictions, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each predicted cell consist of x, y, cls, prob.
    gt_all: List[List[Tuple(int, int, int)]]
        List of GTs, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each GT cell consist of x, y, cls.

    Returns
    -------
    List[List[Tuple(int, np.array, np.array)]]
        Distance (between pred and GT) and Confidence per class and sample.
    """

    all_sample_result = []
    for pred, gt in zip(pred_all, gt_all):
        one_sample_result = {}
        for cls_idx in ALL_CLS_IDX:
            pred_cls = np.array([p for p in pred if p[2] == cls_idx], np.float32)
            gt_cls = np.array([g for g in gt if g[2] == cls_idx], np.float32)

            pred_loc = pred_cls[:, :2].reshape([-1, 1, 2])
            gt_loc = gt_cls.reshape([1, -1, 2])
            distance = np.linalg.norm(pred_loc - gt_loc, axis=2)
            confidence = pred_cls[:, 2]
            one_sample_result[cls_idx] = (distance, confidence)

        all_sample_result.append(one_sample_result)

    return all_sample_result


def _calc_scores(all_sample_result, cls_idx, cutoff):
    """ Calculate Precision, Recall, and F1 scores 
    
    Parameters
    ----------
    all_sample_result: List[List[Tuple(int, np.array, np.array)]]
        Distance (between pred and GT) and Confidence per class and sample.
    cls_idx: int
        1 or 2, where 1 and 2 corresponds Tumor (TC) and Background (BC) cells.
    cutoff: int
        Distance cutoff that used as a threshold for collecting candidates of 
        matching ground-truths per each predicted cell.

    Returns
    -------
    List[List[Tuple(int, np.array, np.array)]]
        Distance (between pred and GT) and Confidence per class and sample.
    """
    
    global_num_gt = 0
    global_num_tp = 0
    global_num_fp = 0

    for one_sample_result in all_sample_result:
        distance, confidence = one_sample_result[cls_idx]
        num_pred, num_gt = distance.shape
        assert len(confidence) == num_pred
        global_num_gt += num_gt

        if num_pred == 0:
            continue
        
        sorted_pred_indices = np.argsort(-confidency)
        bool_mask = (distance <= cutoff)

        num_tp = 0
        num_fp = 0
        for pred_idx in sorted_pred_indices:
            gt_neighbors = bool_mask[pred_idx].nonzero()[0]
            if len(gt_neighbors) == 0:
                # No matching GT --> False Positive
                num_fp += 1
            else:
                # Assign neares GT --> True Positive
                gt_idx = min(gt_neighbors, key=lambda gt_idx: distance[pred_idx, gt_idx])
                num_tp += 1
                bool_mask[:, gt_idx] = False

        assert num_tp + num_fp == num_pred
        global_num_tp += num_tp
        global_num_fp += num_fp
    
    precision = global_num_tp / (global_num_tp + global_num_fp)
    recall = global_num_tp / global_num_gt
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def main(pred_all, gt_all):
    """ Calculate mF1 score and save scores.

    Parameters
    ----------
    pred_all: List[List[Tuple(int, int, int, float)]]
        List of predictions, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each predicted cell consist of x, y, cls, prob.
    gt_all: List[List[Tuple(int, int, int)]]
        List of GTs, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each GT cell consist of x, y, cls.

    Returns
    -------
    float
        A mF1 value which is average of F1 scores of BC and TC classes.
    """

    _check_validity(pred_all, gt_all)
    all_sample_result = _preprocess_distance_and_confidence(pred_all, gt_all)
    scores = {}

    for c in ALL_CLS_IDX:
        precision, recall, f1 = _calc_scores(all_sample_result, c, DISTANCE_CUTOFF)
        scores[c] = {
            "Precision": precision, 
            "Recall": recall, 
            "F1": f1,
        }

    mf1 = sum([scores[c]["F1"] for c in ALL_CLS_IDX]) / len(ALL_CLS_IDX)
    return mF1, scores
