import re
import numpy as np


### These are fixed, don't change!
DISTANCE_CUTOFF = 15
CLS_IDX_TO_NAME = {1: "BC", 2: "TC"}


def _check_validity(inp):
    """ Check validity of algorithm output.

    Parameters
    ----------
    inp: List[Dict]
        List of cell predictions, each element corresponds a cell point.
        Each element is a dictionary with 3 keys, `name`, `point`, `probability`.
        Value of `name` key is `image_{idx}` where `idx` indicates the image index.
        Value of `point` key is a list of three elements, x, y, and cls.
        Value of `probability` key is a confidence score of a predicted cell.
    """
    for cell in inp:
        assert sorted(list(cell.keys())) == ["name", "point", "probability"]
        assert re.fullmatch(r'image_[0-9]+', cell["name"]) is not None
        assert type(cell["point"]) is list and len(cell["point"]) == 3
        assert type(cell["point"][0]) is int and 0 <= cell["point"][0] <= 1023
        assert type(cell["point"][1]) is int and 0 <= cell["point"][1] <= 1023
        assert type(cell["point"][2]) is int and cell["point"][2] in (1, 2)
        assert type(cell["probability"]) is float and 0.0 <= cell["probability"] <= 1.0


def _convert_format(pred_json, gt_json, num_images):
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
    
    num_images: int
        Number of images.
    
    Returns
    -------
    pred_after_convert: List[List[Tuple(int, int, int, float)]]
        List of predictions, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each predicted cell consist of x, y, cls, prob.
    
    gt_after_convert: List[List[Tuple(int, int, int, float)]]
        List of GT, each element corresponds a patch.
        Each patch contains list of tuples, each element corresponds a single cell.
        Each GT cell consist of x, y, cls, prob (always 1.0).
    """
    
    pred_after_convert = [[] for _ in range(num_images)]
    for pred_cell in pred_json:
        x, y, c = pred_cell["point"]
        prob = pred_cell["probability"]
        img_idx = int(pred_cell["name"].split("_")[-1])
        pred_after_convert[img_idx].append((x, y, c, prob))

    gt_after_convert = [[] for _ in range(num_images)]
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
    all_sample_result: List[List[Tuple(int, np.array, np.array)]]
        Distance (between pred and GT) and Confidence per class and sample.
    """

    all_sample_result = []

    for pred, gt in zip(pred_all, gt_all):
        one_sample_result = {}

        for cls_idx in sorted(list(CLS_IDX_TO_NAME.keys())):
            pred_cls = np.array([p for p in pred if p[2] == cls_idx], np.float32)
            gt_cls = np.array([g for g in gt if g[2] == cls_idx], np.float32)
            if len(gt_cls) == 0:
                gt_cls = np.zeros(shape=(0, 4))
            
            if len(pred_cls) == 0:
                distance = np.zeros([0, len(gt_cls)])
                confidence = np.zeros([0, len(gt_cls)])
            else:
                pred_loc = pred_cls[:, :2].reshape([-1, 1, 2])
                gt_loc = gt_cls[:, :2].reshape([1, -1, 2])
                distance = np.linalg.norm(pred_loc - gt_loc, axis=2)
                confidence = pred_cls[:, 2]

            one_sample_result[cls_idx] = (distance, confidence)

        all_sample_result.append(one_sample_result)

    return all_sample_result


def _calc_scores(all_sample_result, cls_idx, cutoff):
    """ Calculate Precision, Recall, and F1 scores for given class 
    
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
    precision: float
        Precision of given class

    recall: float
        Recall of given class

    f1: float
        F1 of given class
    """
    
    global_num_gt = 0
    global_num_tp = 0
    global_num_fp = 0

    for one_sample_result in all_sample_result:
        distance, confidence = one_sample_result[cls_idx]
        num_pred, num_gt = distance.shape
        assert len(confidence) == num_pred

        sorted_pred_indices = np.argsort(-confidence)
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
        global_num_gt += num_gt
        global_num_tp += num_tp
        global_num_fp += num_fp
        
    precision = global_num_tp / (global_num_tp + global_num_fp + 1e-7)
    recall = global_num_tp / (global_num_gt + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def main():
    """ Calculate mF1 score and save scores.

    Returns
    -------
    float
        A mF1 value which is average of F1 scores of BC and TC classes.
    """

    # Path where algorithm output is stored
    algorithm_output_path = "cell_classification.json"
    with open(algorithm_output_path, "r") as f:
        pred_json = json.load(f)["points"]
    
    # Path where GT is stored
    gt_path = "cell_gt.json"
    with open(gt_path, "r") as f:
        gt_json = json.load(f)["points"]
        num_images = json.load(f)["num_images"]

    # Check the validity (e.g. type) of algorithm output
    _check_validity(pred_json)
    _check_validity(gt_json)

    # Convert the format of GT and pred for easy score computation
    pred_all, gt_all = _convert_format(pred_json, gt_json, num_images)

    # For each sample, get distance and confidence by comparing prediction and GT
    all_sample_result = _preprocess_distance_and_confidence(pred_all, gt_all)

    # Calculate scores of each class, then get final mF1 score
    scores = {}
    for cls_idx, cls_name in CLS_IDX_TO_NAME.items():
        precision, recall, f1 = _calc_scores(all_sample_result, cls_idx, DISTANCE_CUTOFF)
        scores[f"Pre/{cls_name}"] = precision
        scores[f"Rec/{cls_name}"] = recall
        scores[f"F1/{cls_name}"] = f1
    
    scores["mF1"] = sum([
        scores[f"F1/{cls_name}"] for cls_name in CLS_IDX_TO_NAME.values()
    ]) / len(CLS_IDX_TO_NAME)
    
    print(scores)
