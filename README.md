# OCELOT23: The algorithm

In this repository you will find the source code for the Grand Challenge OCELOT23 algorithm container.


# Input and output

* The input

* The output
```json
{
    "type": "Multiple points",
    "points": [
        {
            "name": "0",
            "point": [
                128.0,
                620.0,
                1.0
            ],
            "probability": 1.0
        },
        {
            "name": "0",
            "point": [
                128.0,
                621.0,
                1.0
            ],
            "probability": 1.0
        },
```
# Develop you algorithm

At `user/inference.py` you will find the example code to be updated

```python
def process_patch_pair(cell_patch, tissue_patch, pair_id, meta_dataset):
    """This function detects the cells in the cell patch, while additionally
    providing the broader tissue context

    NOTE: this function offers a dummy example inference code. This must be
    updated by the participant.

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
    meta_pair = meta_dataset[pair_id]

    #############################################
    #### YOUR INFERENCE ALGORHTM GOES HERE ######
    #############################################

    # The following is a dummy cell detection algoritm
    prediction = np.copy(cell_patch[2, :, :])
    prediction[(cell_patch[2, :, :] <= 40)] = 1
    xs, ys = np.where(prediction.transpose() == 1)
    probs = [1.0] * len(xs) # Confidence score
    class_id = [1] * len(xs) # Type of cell

    #############################################
    ####### RETURN RESULS PER SAMPLE ############
    #############################################

    # We need to return a list of tuples with 4 elements, i.e.:
    # - cell's x-coordinate in the cell patch
    # - cell's y-coordinate in the cell patch
    # - class id of the cell, either 1 (BC) or 2 (TC)
    # - confidence score of the predicted cell
    return list(zip(xs, ys, class_id, probs))
```

# Submitting to GC

## Build your docker image

```bash
bash build.sh
```
## Testing before submitting to GC

```bash
bash test.sh
```
## Export algorithm docker image

```bash
bash export.sh
```

# Cite


# References