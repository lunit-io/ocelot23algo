import re
import numpy as np


DISTANCE_CUTOFF = 25
ALL_CLS_IDX = (1, 2)


def _check_validity(algorithm_output):
    """ Check validity of algorithm output.

    Parameters
    ----------
    algorithm_output: List[Dict]
        List of cell predictions, each element corresponds a cell point.
        Each element is a dictionary with 3 keys, `name`, `point`, `probability`.
        Value of `name` key is `image_{idx}` where `idx` indicates the image index.
        Value of `point` key is a list of three elements, x, y, and cls.
        Value of `probability` key is a confidence score of a predicted cell.
    """
    for cell in algorithm_output:
        assert all([k in cell for k in ("name", "point", "probability")])
        assert re.fullmatch(r'image_[0-9]+', cell["name"]) is not None
        assert type(cell["point"]) is list and len(cell["point"]) == 3
        assert type(cell["point"][0]) is int and 0 <= cell["point"][0] <= 1023
        assert type(cell["point"][1]) is int and 0 <= cell["point"][1] <= 1023
        assert type(cell["point"][2]) is int and cell["point"][2] in ALL_CLS_IDX
        assert type(cell["probability"]) is float and 0.0 <= cell["probability"] <= 1.0


def _convert_format(pred_json, gt_json):
    """ Helper function that converts the format for easy score computation.

    Parameters
    ----------
    pred_json: List[Dict]
        List of cell predictions, each element corresponds a cell point.
        Each element is a dictionary with 3 keys, `name`, `point`, `probability`.
        Value of `name` key is `image_{idx}` where `idx` indicates the image index.
        Value of `point` key is a list of three elements, x, y, and cls.
        Value of `probability` key is a confidence score of a predicted cell.
    
    gt_json: List[Dict]
        List of cell ground-truths, each element corresponds a cell point.
        Each element is a dictionary with 3 keys, `name`, `point`, `probability`.
        Value of `name` key is `image_{idx}` where `idx` indicates the image index.
        Value of `point` key is a list of three elements, x, y, and cls.
        Value of `probability` key is always 1.0.
    
    Returns
    -------
    List[List[Tuple(int, int, int, float)]]
        List of predictions, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each predicted cell consist of x, y, cls, prob.
    
    List[List[Tuple(int, int, int, float)]]
        List of GT, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each GT cell consist of x, y, cls, prob (always 1.0).
    """

    img_indices = sorted(list(set([int(gt_cell["name"].split("_")[-1]) for gt_cell in gt_json])))
    assert img_indices == list(range(len(img_indices)))

    pred_after_convert = [[]] * len(img_indices)
    for pred_cell in pred_json:
        x, y, c = pred_cell["point"]
        prob = pred_cell["probability"]
        img_idx = int(pred_cell["name"].split("_")[-1])
        pred_after_convert[img_idx].append((x, y, c, prob))

    gt_after_convert = [[]] * len(img_indices)
    for gt_cell in gt_json:
        x, y, c = gt_cell["point"]
        prob = gt_cell["probability"]
        img_idx = int(gt_cell["name"].split("_")[-1])
        gt_after_convert[img_idx].append((x, y, c, prob))

    return pred_after_convert, gt_after_convert


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
        1 or 2, where 1 and 2 corresponds Tumor (TC) and Background (BC) cells, respectively.
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
            if len(gt_neighbors) == 0:  # No matching GT --> False Positive
                num_fp += 1
            else:  # Assign neares GT --> True Positive
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


def main():
    """ Calculate mF1 score and save scores.

    Returns
    -------
    float
        A mF1 value which is average of F1 scores of BC and TC classes.
    """

    # Path where algorithm output is stored
    algorithm_output_path = "cell_predictions.json"
    with open(algorithm_output_path, "r") as f:
        pred_json = json.load(f)["points"]
    
    # Path where GT is stored
    gt_path = "cell_gt.json"
    with open(gt_path, "r") as f:
        gt_json = json.load(f)["points"]

    # Check the validity (e.g. type) of algorithm output
    _check_validity(pred_json)
    _check_validity(gt_json)

    # Convert the format of GT and pred for easy score computation
    pred_all, gt_all = _convert_format(pred_json, gt_json)

    # For each sample, get distance and confidence by comparing prediction and GT
    all_sample_result = _preprocess_distance_and_confidence(pred_all, gt_all)

    # Calculate scores of each class, then get final mF1 score
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