import numpy as np
import os
import nibabel as nib
from pathlib import Path
import scipy
import matplotlib.pyplot as plt

# TODO: read params from command line
filepath = Path.cwd().joinpath("1-200")
target_pixdim = [1, 1, 1]
chunk_size = 16

# should labels also be resampled or rescale output back to original size before comparison?

n1_img = nib.load(filename="91.img.nii.gz")
print(n1_img)

# Resampling
data_shape = n1_img.header.get_data_shape()

# if chunk size is zero, do the whole image at once
if chunk_size == 0:
    chunk_size = data_shape[2]

# pixel dimensions in millimeters
pix_dim = n1_img.header.get_zooms()

data = n1_img.dataobj
#data = n1_img.get_fdata()

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

print(resamp_shape)
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
fig = plt.figure()

plt.plot(x_space, data[:,0,4], c='k', label='data')
plt.plot(xx, resampled_data[:,0,2], c='g', label='linear')

# ax = fig.add_subplot(projection='3d')
# ax.scatter(x_space, y_space, data[:,:,4], s=60, c='k', label='data')
# ax.plot_wireframe(xx, yy, lin_resampled_data[:,:,4], rstride=3, cstride=3,
#                   alpha=0.4, color='m', label='linear')
# ax.plot_wireframe(xx, yy, cub_resampled_data[:,:,4], rstride=3, cstride=3,
#                   alpha=0.4, color='g', label='cubic')

plt.legend()
plt.show()

# Normalisation

# other things