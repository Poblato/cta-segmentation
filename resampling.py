import numpy as np
import os
import sys
import nibabel as nib
from pathlib import Path
import scipy
import matplotlib.pyplot as plt

# param scheme:
# specify_shape: bool
# target: 3 intgers
# input filepath: string
# output filepath: string
# chunk size (optional): int

num_args = len(sys.argv)

if (num_args < 7) or (num_args > 8):
    print("Usage: python resampling.py bool:specify_shape int:target_x int:target_y int:target_shape_z str:input_filepath str:output_filepath int:chunk_size(optional)")
    sys.exit(1)

try:
    specify_shape = bool(int(sys.argv[1]))
    target_shape = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]
    inputfilepath = Path(sys.argv[5])
    outputfilepath = Path(sys.argv[6])
    if num_args == 8:
        chunk_size = int(sys.argv[7])
    else:
        chunk_size = 0
except ValueError:
    print("Error parsing command line arguments.")
    sys.exit(1)

n1_img = nib.load(filename=inputfilepath)
# print(n1_img)

data_shape = n1_img.header.get_data_shape()

# if chunk size is zero, do the whole image at once
if chunk_size == 0:
    chunk_size = data_shape[2]

# pixel dimensions in millimeters
pix_dim = n1_img.header.get_zooms()

data = n1_img.dataobj
#data = n1_img.get_fdata()

if specify_shape:
    target_pixdim = np.divide(np.multiply(pix_dim, data_shape), target_shape)

# print(target_pixdim)

x_space = np.arange(0,data_shape[0])
y_space = np.arange(0,data_shape[1])

steps = np.divide(target_pixdim, pix_dim)

xx = np.arange(0, data_shape[0] - 1, steps[0])
yy = np.arange(0, data_shape[1] - 1, steps[1])

# calculate size of resampled data
num_chunks = np.int32(np.ceil(data_shape[2] / chunk_size))
resamp_shape = np.int32(np.ceil(np.divide(data_shape, steps)))
samples_per_chunk = np.int32(np.ceil(np.divide(chunk_size, steps[2])))
resamp_shape[2] = np.int32(samples_per_chunk * (num_chunks-1))
resamp_shape[2] += np.int32(np.ceil(np.mod(data_shape[2], chunk_size) / steps[2]))

# print(resamp_shape)
resampled_data = np.zeros(shape=resamp_shape)

for i in np.arange(0, num_chunks):
    before_lb = i*chunk_size
    before_ub = min(before_lb + chunk_size, data_shape[2])
    after_lb = i*samples_per_chunk
    after_ub = min(after_lb + samples_per_chunk, data_shape[2])
    z_space = np.arange(before_lb, before_ub)

    zz = np.arange(before_lb, before_ub, steps[2])
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij')

    interp = scipy.interpolate.RegularGridInterpolator((x_space, y_space, z_space), data[:,:,before_lb:before_ub],
                                method="linear", fill_value=None, bounds_error=True, solver=None, solver_args=None)
    resampled_data[:,:,after_lb:after_ub] = interp((X, Y, Z))

# plot against original
# fig = plt.figure()

# plt.plot(x_space, data[:,0,4], c='k', label='data')
# plt.plot(xx, resampled_data[:,0,4], c='g', label='linear')

# plt.legend()
# plt.show()

# calculate affine from new pixdims
affine = n1_img.affine
affine[0:3,0:3] = np.diag(target_pixdim)
affine[0,0] = -affine[0,0] # flip x dim because image coordiantes are weird

# assemble into new Nifti1Image and save
resampled_img = nib.Nifti1Image(resampled_data, affine=affine, header=n1_img.header)
resampled_img.header.set_data_shape(resamp_shape)
resampled_img.header.set_zooms(target_pixdim)
# print(resampled_img)
nib.save(resampled_img, outputfilepath)
