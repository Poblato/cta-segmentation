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
# data array shape
data_shape = n1_img.header.get_data_shape()

if chunk_size == 0:
    chunk_size = data_shape[2]

# pixel dimensions in millimeters
pix_dim = n1_img.header.get_zooms()

data = n1_img.dataobj
#data = n1_img.get_fdata()

x_space = np.arange(0,data_shape[0])
y_space = np.arange(0,data_shape[1])

steps = np.divide(target_pixdim, pix_dim)
# ystep = target_pixdim[1] / pix_dim[1]
# zstep = target_pixdim[2] / pix_dim[2]

xx = np.arange(0, data_shape[0] - 1, steps[0])
yy = np.arange(0, data_shape[1] - 1, steps[1])

# fixme calculate size of resampled data
resamp_shape = np.int32(np.ceil(np.divide(data_shape, steps)))
samples_per_chunk = np.int32(np.ceil(np.divide(chunk_size, steps[2])))
resamp_shape[2] = np.int32(samples_per_chunk * np.floor(data_shape[2] / chunk_size))
resamp_shape[2] += np.int32(samples_per_chunk * np.floor(np.mod(data_shape[2], chunk_size)))

print(resamp_shape)
lin_resampled_data = np.zeros(shape=resamp_shape)
cub_resampled_data = np.zeros(shape=resamp_shape)

for i in np.arange(0, data_shape[2], chunk_size):
    upper_bound = min(i + chunk_size, data_shape[2])
    z_space = np.arange(i, upper_bound)

    zz = np.arange(i, upper_bound, steps[2])
    X, Y, Z = np.meshgrid(xx, yy, zz, indexing='ij')

    # cub_interp = scipy.interpolate.RegularGridInterpolator((x_space, y_space, z_space), data[:,:,i:upper_bound],
    #                             method="cubic", fill_value=None, bounds_error=True, solver=None, solver_args=None)
    # cub_resampled_data[:,:,i:(i+samples_per_chunk)] = cub_interp((X, Y, Z))

    lin_interp = scipy.interpolate.RegularGridInterpolator((x_space, y_space, z_space), data[:,:,i:upper_bound],
                                method="linear", fill_value=None, bounds_error=True, solver=None, solver_args=None)
    lin_resampled_data[:,:,i:(i+samples_per_chunk)] = lin_interp((X, Y, Z))

# plot against original
fig = plt.figure()

plt.plot(x_space, data[:,4,4], c='k', label='data')
plt.plot(xx, cub_resampled_data[:,4,4], c='m', label='cubic')
plt.plot(xx, lin_resampled_data[:,4,4], c='g', label='linear')

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