# OCELOT23: The algorithm
 
In this repository you will find the source code for the Grand Challenge OCELOT23 algorithm container. Ocelot is both a MICCAI challenge and an accepeted paper at CVPR 23.
 
# Input and output
 
* The input: the container searches loads and iterates over the validation images, test images and metadata from the already uploaded data in Grand Challenge. The implemented loader `DataLoader` at `util.gcio.py` will iterate over the samples for you!. 

* The output: your algorithm needs to predict cells with the Multiple Points format. To make things easier like with the data loader, we implemented a simple writer class `DetectionWriter` to output the corresponding output file `cell_predictions.json`. An example of the output can be found in `test/output/example_output.json`.

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

At `user/inference.py` you will find a dummy cell detection algorithm. Your task is to modify the function `process_patch_pair` trying to keep the format used below. Feel free to install any framework, such as PyTorch or Tensorflow to run your code. In addition, do not forget to add your dependencies in `requirement.txt` so that your container can be build correctly.

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
    #### YOUR INFERENCE ALGORITHM GOES HERE #####
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

To submit your algorithm to the GC platform, you'll need to export the docker container wrapping your code.

### Build your docker image

```bash
bash build.sh
```

### Testing before submitting to GC

Before submitting your containers to GC, make sure this simple test works in your local machine. This script will create the image, run the container and verify that the output `cell_predictions.json` at the output directory. To do so, run the following command:

```bash
bash test.sh
```
### Export algorithm docker image

Generate the file to be uploaded to GC by running the following script:

```bash
bash export.sh
```

# Cite
