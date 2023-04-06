import tifffile
import glob
import os
import imageio
from PIL import Image
import numpy as np

from gcio import TIFFIterator

# Example usage
ROOT = "ocelot2023_v1.0.0"
SPLITS = ["test", "val"]
TARGET_FOLDER = "images"
TASKS = ["tissue", "cell"]


def create_tiff_from_png(png_files, tiff_file):
    with tifffile.TiffWriter(tiff_file, bigtiff=True) as tif:
        for png_file in png_files:
            # Read PNG image data
            image_data = Image.open(png_file)
            
            # Convert image to numpy array
            image_data = np.array(image_data)
            
            file_id = os.path.split(png_file)[1]
            assert ".png" in file_id
            file_id = file_id.replace(".png","")

            # Write image data to TIFF file
            tif.save(image_data, description=file_id)

    print(f'TIFF file created: {tiff_file}')


def create_tiff_from_jpeg(jpeg_files, tiff_file):
    with tifffile.TiffWriter(tiff_file, bigtiff=True) as tif:
        for jpeg_file in jpeg_files:
            # Read JPEG image data
            image_data = imageio.imread(jpeg_file)

            file_id = os.path.split(jpeg_file)[1]
            assert ".jpg" in file_id
            file_id = file_id.replace(".jpg","")

            # Write image data to TIFF file
            tif.save(image_data, description=file_id)
    
    print(f'TIFF file created: {tiff_file}')


def export_jpeg():
    for split in SPLITS:
        for task in TASKS:
            path = os.path.join(ROOT, TARGET_FOLDER, split, task)
            file_list = sorted(glob.glob(path + '/*'))
            print(f"{task}-{split} - file list len {len(file_list)}")

            assert len(file_list) > 0

            tiff_file = f'{task}_{split}_output.tif'  # Output TIFF file
            create_tiff_from_jpeg(file_list, tiff_file)


def validate_tiff():
    """ We compare the difference between the tif loaded
    and the original image. We must have 0 difference pixel-wise
    """
    count = 0
    error = 0.0
    for split in SPLITS:
        cell_tif = f"cell_{split}_output.tif" 
        tissue_tif = f"tissue_{split}_output.tif" 
        iterator = TIFFIterator(cell_tif, tissue_tif)
        for cell_patch, tissue_patch, pair_id in iterator:
            root = os.path.join(ROOT, TARGET_FOLDER, split)
            
            or_cell_fpath = os.path.join(root, "cell", f"{pair_id}.jpg")
            or_tissue_fpath = os.path.join(root, "tissue", f"{pair_id}.jpg")
            or_cell_patch = imageio.imread(or_cell_fpath)
            or_tissue_patch = imageio.imread(or_tissue_fpath)
            count += 1
            error += np.abs(or_cell_patch - cell_patch).sum()
            error += np.abs(or_tissue_patch - tissue_patch).sum()

    assert count > 0, "No samples were preprocessed"
    assert error == 0.0, f" There is a mismatch between the exported and the original"
    print(f"Validation completed successfully")

if __name__ == "__main__":
     export_jpeg()
     # validate_tiff()

