import os
import nibabel as nib
import numpy as np

root_dir = "/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2"

global_min = float('inf')
global_max = float('-inf')

# Recursively traverse the directory
for root, dirs, files in os.walk(root_dir):
    for fname in files:
        # Check for .nii.gz extension and skip if 'seg' is in the filename
        if fname.endswith(".nii.gz") and "seg" not in fname.lower():
            file_path = os.path.join(root, fname)
            print(f"Processing: {file_path}")

            # Load the NIfTI image
            img = nib.load(file_path)

            # Get floating-point data
            data = img.get_fdata()

            # Compute min and max for the current file
            curr_min = np.min(data)
            curr_max = np.max(data)

            # Update global min and max
            if curr_min < global_min:
                global_min = curr_min
            if curr_max > global_max:
                global_max = curr_max

# Print the global min and max intensities
print(f"\nGlobal minimum intensity across all images: {global_min}")
print(f"Global maximum intensity across all images: {global_max}")