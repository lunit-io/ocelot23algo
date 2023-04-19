import os
import glob
import json
import argparse


def main(dataset_root_path, subset):
    """ Convert csv annotations into a single JSON and save it,
        to match the format with the algorithm submission output.

    Parameters:
    -----------
    dataset_root_path: str
        path to the dataset. (e.g. /home/user/ocelot2023_v0.1.1)
    
    subset: str
        `train` or `val` or `test`.
    """

    assert os.path.exists(f"{dataset_root_path}/annotations/{subset}")
    gt_paths = sorted(glob.glob(f"{dataset_root_path}/annotations/{subset}/cell/*.csv"))
    num_images = len(gt_paths)

    gt_json = {
        "type": "Multiple points",
        "num_images": num_images,
        "points": [],
        "version": {
            "major": 1,
            "minor": 0,
        }
    }
    
    for idx, gt_path in enumerate(gt_paths):
        with open(gt_path, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            x, y, c = line.split(",")
            point = {
                "name": f"image_{idx}",
                "point": [int(x), int(y), int(c)],
                "probability": 1.0,  # dummy value, since it is a GT, not a prediction
            }
            gt_json["points"].append(point)

    with open(f"cell_gt_{subset}.json", "w") as g:
        json.dump(gt_json, g)
        print(f"JSON file saved in {os.getcwd()}/cell_gt_{subset}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root_path", type=str, required=True,
                        help="Path to the dataset. (e.g. /home/user/ocelot2023_v0.1.1)")
    parser.add_argument("-s", "--subset", type=str, required=True, 
                        choices=["train", "val", "test"],
                        help="Which subset among (trn, val, test)?")
    args = parser.parse_args()
    main(args.dataset_root_path, args.subset)