from util import gcio
from util.constants import (
    GC_CELL_FPATH, 
    GC_TISSUE_FPATH, 
    GC_METADATA_FPATH,
    GC_DETECTION_OUTPUT_PATH
)
from user.inference import process_patch_pair


def process():
    """Process a test patches. This involves iterating over samples,
    inferring and write the cell predictions
    """
    # Initialize the data loader
    loader = gcio.DataLoader(GC_CELL_FPATH, GC_TISSUE_FPATH)

    # Cell detection writer
    writer = gcio.DetectionWriter(GC_DETECTION_OUTPUT_PATH)
    
    # Loading metadata
    meta_dataset = gcio.read_json(GC_METADATA_FPATH)

    # NOTE: Batch size is 1
    for cell_patch, tissue_patch, pair_id in loader:
        print(f"Processing sample pair {pair_id}")
        # Cell-tissue patch pair inference
        cell_predictions = process_patch_pair(cell_patch, 
                                    tissue_patch, 
                                    pair_id,
                                    meta_dataset)
        
        # Updating predictions
        writer.add_points(cell_predictions, pair_id)

    # Export the prediction into a json file
    writer.save()


if __name__ == "__main__":
    process()

