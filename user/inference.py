import numpy as np


def process_patch_pair(cell_patch, tissue_patch, pair_id, meta_dataset):
    """This function detects the cells in the cell patch, while additionally
    providing the broader tissue context

    NOTE: this function offers a dummy example inference code. This must be
    updated by the participant.

    Parameters
    ----------
    cell_patch: np.ndarray[uint8]
        Cell patch with shape [1024, 1024, 3] with values from 0 - 255
    tissue_patch: np.ndarray[uint8] 
        Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
    pair_id: str
        identification number of the patch pair
    meta_dataset: Dict
        Dataset metadata in case you wish to compute statistics

    Returns
    -------
        List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
    """
    # Getting the metadata corresponding to the patch pair ID
    meta_pair = meta_dataset[pair_id]

    #############################################
    #### YOUR INFERENCE ALGORITHM GOES HERE #####
    #############################################

    # The following is a dummy cell detection algorithm
    prediction = np.copy(cell_patch[:, :, 2])
    prediction[(cell_patch[:, :, 2] <= 40)] = 1
    xs, ys = np.where(prediction.transpose() == 1)
    probs = [1.0] * len(xs) # Confidence score
    class_id = [1] * len(xs) # Type of cell

    #############################################
    ####### RETURN RESULS PER SAMPLE ############
    #############################################

    # We need to return a list of tuples with 4 elements, i.e.:
    # - int: cell's x-coordinate in the cell patch
    # - int: cell's y-coordinate in the cell patch
    # - int: class id of the cell, either 1 (BC) or 2 (TC)
    # - float: confidence score of the predicted cell
    return list(zip(xs, ys, class_id, probs))
