# How algorithm is evaluated?
Whenever participants submit the algorithm through the Grand Challenge, inference, and evaluation will be automatically proceeded in the backend with a hidden dataset. 
We provide a high-level explanation of the evaluation process at https://ocelot2023.grand-challenge.org/evaluation-metric/, however, we also provide the actual code in `evaluation/eval.py`. 

# Try evaluation
Follow the below steps to try the evaluation with the training dataset.

1. Save a single JSON file that stores cell predictions by following a format described in https://github.com/lunit-io/ocelot23algo/blob/main/README.md#input-and-output
2. Save a single JSON file that stores ground-truth cells which is originally a list of ground-truth CSV files in [Zenodo](https://zenodo.org/record/7844149#.ZEZtlOxBzIE). It can be easily done by `python convert_gt_csvs_to_json.py -d DATASET_PATH -s train`. This is for matching the format with the JSON for the cell predictions.
3. Properly update `algorithm_output_path` and `gt_path` in the `evaluation/eval.py`.
4. Run `python evaluation/eval.py`

Note that the `evaluation/eval.py` uses exactly the same code as the one embedded in the Grand Challenge for auto-evaluation.