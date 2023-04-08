import tifffile
import cv2
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


def create_tiff_from_jpeg(jpeg_files, tiff_file):
    """Generate tiff with pages from jpeg"""
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


def create_tiff_from_tiffs(tiff_files, target_tiff_file):
    """Generate tiff with pages from tiff"""
    with tifffile.TiffWriter(target_tiff_file, bigtiff=True) as tif:
        for tiff_file in tiff_files:
            # Read TIFF image data
            image_data = Image.open(tiff_file)

            file_id = os.path.split(tiff_file)[1]
            assert ".tiff" in file_id
            file_id = file_id.replace(".tiff","")

            # Write image data to TIFF file
            tif.save(image_data, description=file_id)

    print(f'TIFF file created: {target_tiff_file}')

def concatenate_tiff_from_tiffs(tiff_files, target_tiff_file):
    """Generate tiff with concatenated tiff in the H axis"""
    image_data = None
    with tifffile.TiffWriter(target_tiff_file, bigtiff=True) as tif:
        for tiff_file in tiff_files:
            # Read TIFF image data
            cur_image_data = Image.open(tiff_file)
            
            file_id = os.path.split(tiff_file)[1]
            assert ".tiff" in file_id

            print(f"file_id: {file_id}")
            if image_data is None:
                image_data = np.copy(cur_image_data)
            else:
                image_data = np.concatenate((image_data, np.copy(cur_image_data)), axis=0)
        # Write image data to TIFF file
        tif.save(image_data)
    print(f'TIFF file created: {target_tiff_file}')


def export(format: str):
    for split in SPLITS:
        for task in TASKS:
            path = os.path.join(ROOT, TARGET_FOLDER, split, task)
            file_list = sorted(glob.glob(path + '/*'))
            print(f"sorted files: {file_list}")
            print(f"{task}-{split} - file list len {len(file_list)}")

            assert len(file_list) > 0

            tiff_file = f'{task}_{split}.tif'  # Output TIFF file
            if format == 'jpeg':
                create_tiff_from_jpeg(file_list, tiff_file)
            elif format == 'tif':
                # create_tiff_from_tiffs(file_list, tiff_file)
                concatenate_tiff_from_tiffs(file_list, tiff_file)
            else:
                raise Exception


def validate_tiff_with_jpeg():
    """ We compare the difference between the tif loaded
    and the original image. We must have 0 difference pixel-wise
    """
    count = 0
    error = 0.0
    for split in SPLITS:
        cell_tif = f"cell_patches/cell_{split}.tif" 
        tissue_tif = f"tissue_patches/tissue_{split}.tif" 
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


def validate_pages_tiff():
    """ We compare the difference between the tif loaded
    and the original image. We must have 0 difference pixel-wise
    """
    count = 0
    error = 0.0
    for split in SPLITS:
        cell_tif = f"cell_patches/cell_{split}.tif" 
        tissue_tif = f"tissue_patches/tissue_{split}.tif" 
        iterator = TIFFIterator(cell_tif, tissue_tif)
        for cell_patch, tissue_patch, pair_id in iterator:
            root = os.path.join(ROOT, TARGET_FOLDER, split)
            

            or_cell_fpath = os.path.join(root, "cell", f"{pair_id}.tiff")
            or_tissue_fpath = os.path.join(root, "tissue", f"{pair_id}.tiff")

            or_cell_patch = np.array(Image.open(or_cell_fpath))
            or_tissue_patch = np.array(Image.open(or_tissue_fpath))

            or_cell_patch = np.transpose(or_cell_patch,[2, 0, 1])
            or_tissue_patch = np.transpose(or_tissue_patch, [2, 0, 1])

            count += 1
            error += np.abs(or_cell_patch - cell_patch).sum()
            error += np.abs(or_tissue_patch - tissue_patch).sum()

    assert count > 0, "No samples were preprocessed"
    assert error == 0.0, f" There is a mismatch between the exported and the original"
    print(f"Validation completed successfully")


def validate_concatenated_tiff():
    """ We compare the difference between the tif loaded
    and the original image. We must have 0 difference pixel-wise
    """
    error = 0.0
    for split in SPLITS:
        cell_tif = f"cell_{split}.tif" 
        tissue_tif = f"tissue_{split}.tif" 
        cell_patches = np.array(Image.open(cell_tif))
        tissue_patches = np.array(Image.open(tissue_tif))

        dataset_size = cell_patches.shape[0] // 1024
        assert dataset_size > 0
        for i in range(dataset_size):
            cell_patch = cell_patches[i*1024:(i+1)*1024,:,:]
            tissue_patch = tissue_patches[i*1024:(i+1)*1024,:,:]

            root = os.path.join(ROOT, TARGET_FOLDER, split)

            file_list = sorted(glob.glob(os.path.join(root, "cell") + '/*'))
            pair_id = os.path.split(file_list[i])[1]
            
            or_cell_fpath = os.path.join(root, "cell", f"{pair_id}")
            or_tissue_fpath = os.path.join(root, "tissue", f"{pair_id}")

            or_cell_patch = np.array(Image.open(or_cell_fpath))
            or_tissue_patch = np.array(Image.open(or_tissue_fpath))

            error += np.abs(or_cell_patch - cell_patch).sum()
            error += np.abs(or_tissue_patch - tissue_patch).sum()

    assert error == 0.0, f" There is a mismatch between the exported and the original"
    print(f"Validation completed successfully")

if __name__ == "__main__":
    export('tif')
    # validate_concatenated_tiff()

