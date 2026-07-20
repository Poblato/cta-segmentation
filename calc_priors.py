import nibabel as nib
from pathlib import Path
import numpy as np

# average proportions over 10 images
num_images = 200

pos_p = 0.0

for i in range(num_images):
    lbl = nib.load(Path(f"dataset/processed/labels/{i+1}.label_processed_norm.nii.gz"))
    data = lbl.dataobj
    pos_p += np.mean(data) / num_images

print(pos_p)
