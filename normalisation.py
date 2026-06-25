import sys
import numpy as np
import nibabel as nib
from pathlib import Path

# param scheme:
# mean: float
# std_dev: float
# input filepath: string
# output filepath: string

num_args = len(sys.argv)

if (num_args != 6):
    print("Usage: python normalisation.py float:mean, float:std_dev str:input_filepath str:output_filepath")
    sys.exit(1)

try:
    target_mean = np.float64(sys.argv[1])
    target_std_dev = np.float64(sys.argv[2])
    inputfilepath = Path(sys.argv[3])
    outputfilepath = Path(sys.argv[4])
except ValueError:
    print("Error parsing command line arguments.")
    sys.exit(1)

# target_mean = 1
# target_std_dev = 0.25
# inputfilepath = "91_sampled.img.nii.gz"
# outputfilepath = "91_sampled_normalized.img.nii.gz"

n1_img = nib.load(filename=inputfilepath)
data_shape = n1_img.header.get_data_shape()
data = n1_img.dataobj

# Calculate mean and standard deviation
mean_val = np.mean(data)
std_dev_val = np.std(data)

# Avoid division by zero if std_dev is 0
if std_dev_val == 0:
    normalised_image = np.zeros_like(data_shape)
else:
    normalised_image = (data - (target_mean - mean_val)) / (std_dev_val/target_std_dev)

# assemble into new Nifti1Image and save
normalised_n1b = nib.Nifti1Image(normalised_image, affine=n1_img.affine, header=n1_img.header)
normalised_n1b.header.set_data_shape(data_shape)
# print(normalised_image)
nib.save(normalised_n1b, outputfilepath)