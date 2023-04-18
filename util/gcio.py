import os
from pathlib import Path
import numpy as np
from PIL import Image
import json
from typing import List

SAMPLE_SHAPE = (1024,1024,3)


def read_json(fpath: Path) -> dict:
    """This function reads a json file

    Parameters
    ----------
    fpath: Path
        path to the json file

    Returns
    -------
    dict:
        loaded data 
    """
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data


class DataLoader:
    def __init__(self, cell_path, tissue_path):
        cell_fpath = [os.path.join(cell_path, f) for f in os.listdir(cell_path) if ".tif" in f]
        tissue_fpath = [os.path.join(tissue_path, f) for f in os.listdir(tissue_path) if ".tif" in f]
        assert len(cell_fpath) == len(tissue_fpath) == 1

        self.cell_patches = np.array(Image.open(cell_fpath[0]))
        self.tissue_patches = np.array(Image.open(tissue_fpath[0]))

        assert (self.cell_patches.shape[1:] == SAMPLE_SHAPE[1:]), \
            "The same of the input cell patch is incorrect"
        assert (self.tissue_patches.shape[1:] == SAMPLE_SHAPE[1:]), \
            "The same of the input tissue patch is incorrect"

        # Samples are concatenated across the first axis
        assert self.cell_patches.shape[0] % SAMPLE_SHAPE[0] == 0
        assert self.tissue_patches.shape[0] % SAMPLE_SHAPE[0] == 0

        self.num_images = self.cell_patches.shape[0] // SAMPLE_SHAPE[0]

        assert self.num_images == self.tissue_patches.shape[0]//SAMPLE_SHAPE[0], \
            "Cell and tissue patches have different number of instances"

        self.cur_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_idx < self.num_images:
            # Read patch pair and the corresponding id 
            cell_patch = self.cell_patches[self.cur_idx*SAMPLE_SHAPE[0]:(self.cur_idx+1)*SAMPLE_SHAPE[0],:,:]
            tissue_patch = self.tissue_patches[self.cur_idx*SAMPLE_SHAPE[0]:(self.cur_idx+1)*SAMPLE_SHAPE[0],:,:]

            pair_id = self.cur_idx

            # Increment the current image index
            self.cur_idx += 1

            # Return the image data
            return cell_patch, tissue_patch, pair_id
        else:
            # Raise StopIteration when no more images are available
            raise StopIteration


class DetectionWriter:
    """ This class writes the cell predictions to the designated 
    json file path with the Multiple Point format required by 
    Grand Challenge

    Parameters
    ----------
    output: Path
        output_path (Path): path to json output file
    """

    def __init__(self, output_path: Path):

        if output_path.suffix != '.json':
            output_path = output_path / '.json' 

        self._output_path = output_path
        self._data = {
            "type": "Multiple points",
            "points": [],
            "version": {"major": 1, "minor": 0},
        } 

    def add_point(
            self, 
            x: int, 
            y: int,
            class_id: int,
            prob: float, 
            sample_id: int
        ):

        point = {
            "name": "image_{}".format(str(sample_id)),
            "point": [int(x), int(y), int(class_id)],
            "probability": prob}
        self._data["points"].append(point)

    def add_points(self, points: List, sample_id: str):
        for x, y, c, prob in points:
            self.add_point(x, y, c, prob, sample_id)

    def save(self):
        assert len(self._data["points"]) > 0, "No cells were predicted"
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=4)
        print(f"Predictions were saved at `{self._output_path}`")

