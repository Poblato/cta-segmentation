import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from medpy.filter.smoothing import anisotropic_diffusion

# param scheme:
# niter: int
# kappa: float
# gamma: float
# input filepath: string
# output filepath: string

num_args = len(sys.argv)

if (num_args != 6):
    print("Usage: python noise_reduction.py int:niter float:kappa float:gamma str:input_filepath str:output_filepath")
    sys.exit(1)

try:
    niter = int(sys.argv[1])
    kappa = np.float64(sys.argv[2])
    gamma = np.float64(sys.argv[3])
    inputfilepath = Path(sys.argv[4])
    outputfilepath = Path(sys.argv[5])
except ValueError:
    print("Error parsing command line arguments.")
    sys.exit(1)

# niter = 1
# kappa = 25
# gamma = 0.1
# inputfilepath = "91_sampled_normalised.img.nii.gz"
# outputfilepath = "91_sampled_normalised_denoised.img.nii.gz"

image = nib.load(inputfilepath)
data = image.dataobj
data_shape = image.header.get_data_shape()

# apply anisotropic diffusion filter
filtered_image = anisotropic_diffusion(data, niter=niter, kappa=kappa, gamma=gamma, voxelspacing=image.header.get_zooms())

# assemble into new Nifti1Image and save
filtered_n1b = nib.Nifti1Image(filtered_image, affine=image.affine, header=image.header)
filtered_n1b.header.set_data_shape(data_shape)
# print(filtered_image)
nib.save(filtered_n1b, outputfilepath)