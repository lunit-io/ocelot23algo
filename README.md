# OCELOT23: The algorithm
 
In this repository, you can find the source code for the [Grand Challenge OCELOT 23](https://ocelot2023.grand-challenge.org/) algorithm container. We highly recommend using this repository as template for your algorithm submissions. For more information refer to our [page](https://lunit-io.github.io/research/publications/ocelot/).

 
# Input and output
 
We already implemented for you the input/output interface for loading the input images stored in the platform and writing the cell predictions. Here the relevant code:
* The input: the container loads and iterates over the validation images, test images and metadata from the already uploaded data in Grand Challenge (not visible to partipants). The implemented data loader `DataLoader` at `util.gcio.py` will iterate over the samples for you!. 

* The output: your algorithm needs to predict cells with the [Multiple Points](https://comic.github.io/grand-challenge.org/components.html#grandchallenge.components.models.InterfaceKind.interface_type_json) format. To make things easier, we developed a simple writer class `DetectionWriter` to output the corresponding output file `cell_predictions.json`. An example of the output can be found in `test/output/example_output.json`.

```json
{
    "type": "Multiple points",
    "points": [
        {
            "name": "image_0",
            "point": [
                128,
                620,
                1
            ],
            "probability": 1.0
        },
        {
            "name": "image_0",
            "point": [
                128,
                621,
                1
            ],
            "probability": 1.0
        },
```
Where each cell prediction requires the following information:

* `name`: cell patch identifier, which is composed of the keyword `image` followed by the sequential image ID of the cell patch. The ID is the same as the one provided by the `DataLoader`.
* `point`: list of three intiger, i.e. x, y and class ID.
* `probability`: confidence score of the predicted cell.

# Develop you algorithm

At `user/inference.py` you will find a dummy cell detection algorithm. Your task is to propose a new algorithm by modifying the function `process_patch_pair` while keeping the returned format used below. Feel free to install any framework, such as PyTorch or Tensorflow by adding your dependencies in `requirements.txt`.

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

To submit your algorithm to the GC platform, you'll need to export the docker container with your all the required elements to run inference on the patches.

### Build your docker image

Build your image with the next command:

```bash
bash build.sh
```

### Testing before submitting to GC

Before submitting your containers to GC, make sure the proposed test works successfully in your local machine. The script `test.sh` will create the image, run a container and verify that the output file `cell_predictions.json` is generated at the designated output directory. To do so, simply run the following command:

```bash
bash test.sh
```

### Export algorithm docker image

Generate the `tar` file to be uploaded to GC by with the command:

```bash
bash export.sh
```

# Citation

Cite our work!
```
@misc{ryu2023ocelot,
      title={OCELOT: Overlapped Cell on Tissue Dataset for Histopathology}, 
      author={Jeongun Ryu and Aaron Valero Puche and JaeWoong Shin and Seonwook Park and Biagio Brattoli and Jinhee Lee and Wonkyung Jung and Soo Ick Cho and Kyunghyun Paeng and Chan-Young Ock and Donggeun Yoo and Sérgio Pereira},
      year={2023},
      eprint={2303.13110},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

**NOTE**: We will update the citation when CVPR publicly release the proceedings!