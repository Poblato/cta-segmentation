import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.filters import frangi

# param scheme:
# sigma_min: int
# sigma_max: int
# sigma_step: int
# alpha: float
# beta: float
# gamma: float
# input filepath: string
# output filepath: string

num_args = len(sys.argv)

if (num_args != 9):
    print("Usage: python frangi_filter.py int:sigma_min int:sigma_max int:sigma_step float:alpha float:beta float:gamma str:input_filepath str:output_filepath")
    sys.exit(1)

try:
    sigma_min = int(sys.argv[1])
    sigma_max = int(sys.argv[2])
    sigma_step = int(sys.argv[3])
    alpha = np.float64(sys.argv[4])
    beta = np.float64(sys.argv[5])
    gamma = np.float64(sys.argv[6])
    inputfilepath = Path(sys.argv[7])
    outputfilepath = Path(sys.argv[8])
except ValueError:
    print("Error parsing command line arguments.")
    sys.exit(1)

image = nib.load(inputfilepath)
data = image.dataobj
data_shape = image.header.get_data_shape()


# Apply the Frangi filter with the default values
filtered_image = frangi(data, sigmas=np.arange(sigma_min, sigma_max, sigma_step), alpha=alpha, beta=beta, gamma=gamma, black_ridges=False, mode='reflect')

# assemble into new Nifti1Image and save
filtered_n1b = nib.Nifti1Image(filtered_image, affine=image.affine, header=image.header)
filtered_n1b.header.set_data_shape(data_shape)
# print(filtered_image)
nib.save(filtered_n1b, outputfilepath)