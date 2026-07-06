import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

img = nib.load("dataset/processed/labels/1.label_processed_norm.nii.gz")

print(img)

plt.figure()
plt.imshow(img.dataobj[:, :, 64], cmap="gray")
plt.show()
