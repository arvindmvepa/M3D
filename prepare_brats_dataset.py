import os
import numpy as np
from PIL import Image
import concurrent.futures
from tqdm import tqdm
from collections import Counter
import unicodedata
import monai.transforms as mtf
from multiprocessing import Pool
from unidecode import unidecode
import nibabel as nib


#input_dir = '/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2'
#output_dir = '/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2_npy'
input_dir = '/local2/shared_data/BraTS2024-BraTS-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth'
output_dir = '/local2/shared_data/BraTS2024-BraTS-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth_npy'

# Get all subfolders [00001, 00002....]
subfolders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]


transform = mtf.Compose([
    mtf.CropForeground(),
    mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")
])


def process_subfolder(subfolder):
    output_id_folder = os.path.join(output_dir, subfolder)
    input_id_folder = os.path.join(input_dir, subfolder)

    os.makedirs(output_id_folder, exist_ok=True)

    dir_files = [file for file in os.listdir(input_id_folder) if file.endswith('.nii.gz')]
    image_files = [file for file in dir_files if 'seg' not in file]
    assert len(image_files) > 0, f"No image files found in {subsubfolder_path}"

    for image_file in image_files:
        image_path = os.path.join(input_id_folder, image_file)
        output_path = os.path.join(output_id_folder, f'{image_file}.npy')
        try:
            img = nib.load(image_path)
            img_array = img.get_fdata()
            # normalization (unnecessary because min/max normalization is done below)
            # img_array = (img_array.astype(np.float32) - min_intensity) / (max_intensity - min_intensity)
        except:
            print("This image is error: ", image_path)

        try:
            image = img_array[np.newaxis, ...]

            image = image - image.min()
            image = image / np.clip(image.max(), a_min=1e-8, a_max=None)

            img_trans = transform(image)

            np.save(output_path, img_trans)
        except:
            print([img.shape for img in images_3d])
            print("This folder is vstack error: ", output_path)


with Pool(processes=32) as pool:
    with tqdm(total=len(subfolders), desc="Processing") as pbar:
        for _ in pool.imap_unordered(process_subfolder, subfolders):
            pbar.update(1)
