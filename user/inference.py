import numpy as np
from typing import List, Dict


def process_patch_pair(cell_patch, tissue_patch, pair_id, meta_dataset):
    """This function detects the cells in the cell patch, while additionally
    providing the broader tissue context

    NOTE: this function offers a dummy example inference code. This must be
    updated by the user.

    Parameters
    ----------
    cell_patch: np.ndarray 
        Cell patch with shape [3, 1024, 1024]
    tissue_patch: np.ndarray 
        Tissue patch with shape [3, 1024, 1024]
    pair_id: str
        identification number of the patch pair
    meta_dataset: Dict
        Dataset metadata in case you wish to compute statistics

    Returns
    -------
        List[tuple]: list of tuples (x,y) coordinates of detections
    """
    # Getting the metadata corresponding to the patch pair ID
    meta_pair = meta_dataset["sample_pairs"][pair_id]

    ############################################# 
    ##### THE INFERENCE ALGORHTM GOES HERE ######
    #############################################

    # The following is a dummy cell detection algoritm
    prediction = np.copy(cell_patch[2, :, :])
    prediction[(cell_patch[2, :, :] <= 40)] = 1
    xs, ys = np.where(prediction.transpose() == 1)
    probs = [1.0] * len(xs) # Confidence score
    class_id = [1] * len(xs) # Type of cell

    # We need to return a list of tuples with 4 elements, i.e.:
    # - cell's x-coordinate in the cell patch
    # - cell's y-coordinate in the cell patch
    # - class id of the cell, either 1 (BC) or 2 (TC)
    # - confidence score of the predicted cell
    return list(zip(xs, ys, class_id, probs))
