from pathlib import Path
import numpy as np
import tifffile
import json
from typing import Union, List


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


class TIFFIterator:
    def __init__(self, cell_fpath, tissue_fpath):
        self.tissue_fpath = cell_fpath
        self.cell_fpath = cell_fpath
        self.cell_tiff = tifffile.TiffFile(cell_fpath)
        self.tissue_tiff = tifffile.TiffFile(tissue_fpath)

        self.num_images = len(self.cell_tiff.pages)
        assert len(self.cell_tiff.pages) == len(self.tissue_tiff.pages)

        self.cur_patch_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_patch_idx < self.num_images:
            # Read patch pair and the corresponding id 
            cell_patch = self.cell_tiff.pages[self.cur_patch_idx].asarray()
            cell_patch = np.rollaxis(cell_patch, 2, 0)
            cell_id = self.cell_tiff.pages[self.cur_patch_idx].description

            tissue_patch = self.tissue_tiff.pages[self.cur_patch_idx].asarray()
            tissue_patch = np.rollaxis(tissue_patch, 2, 0)
            tissue_id = self.tissue_tiff.pages[self.cur_patch_idx].description

            # We assume the tissue and cell patches are saved aligned
            assert cell_id == tissue_id

            # Increment the current image index
            self.cur_patch_idx += 1

            # Return the image data
            return cell_patch, tissue_patch, cell_id
        else:
            # Raise StopIteration when no more images are available
            raise StopIteration


class DetectionWriter:
    """Writes detection to json format that can be handled by grand challenge"""

    def __init__(self, output_path: Path):
        """init

        Args:
            output_path (Path): path to json output file
        """

        if output_path.suffix != '.json':
            output_path = output_path / '.json' 

        self._output_path = output_path
        self._data = {}
        self._cur_sample = {}

    def add_point(self, x:  Union[int, float], y:  Union[int, float], prob: float, sample_id: str):
        Z = 0.5
        point = {"point": [float(x), float(y), Z], "probability": prob}
        if sample_id not in self._data:
            multiple_point = {
                "type": "Multiple points",
                "points": [],
                "version": {"major": 1, "minor": 0},
            } 
            self._data[sample_id] = multiple_point
        self._data[sample_id]["points"].append(point)

    def add_points(self, points: List, sample_id: str):
        for x, y, prob in points:
            self.add_point(x, y, prob, sample_id)

    def save(self):
        assert len(self._data) > 0, "No cells were predicted"
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=4)
        print(f"Predictions were saved at `{self._output_path}`")

