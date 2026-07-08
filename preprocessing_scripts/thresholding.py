import sys
import numpy as np
import nibabel as nib
from pathlib import Path

# param scheme:
# threshold: float
# input filepath: string
# output filepath: string

num_args = len(sys.argv)

if (num_args != 4):
    print("Usage: python thresholding.py float:threshold str:input_filepath str:output_filepath")
    sys.exit(1)

try:
    threshold = np.float64(sys.argv[1])
    inputfilepath = Path(sys.argv[2])
    outputfilepath = Path(sys.argv[3])
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
thresholded_image = np.float64(data > threshold)

# assemble into new Nifti1Image and save
thresholded_n1b = nib.Nifti1Image(thresholded_image, affine=n1_img.affine, header=n1_img.header)
thresholded_n1b.header.set_data_shape(data_shape)
# print(thresholded_image)
nib.save(thresholded_n1b, outputfilepath)