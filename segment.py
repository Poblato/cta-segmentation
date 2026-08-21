import os
import sys
import torch
import numpy as np
import segmentation_models_pytorch as smp
import nibabel as nib
import torchio as tio
from pathlib import Path
from WeightedLoss import WeightedLoss
import matplotlib.pyplot as plt

num_args = len(sys.argv)

if (num_args != 5):
    print("Usage: python segment.py str:model_filepath str:image_filepath str:output_filepath bool:vis")
    sys.exit(1)

try:
    model_path = Path(sys.argv[1])
    image_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    vis = bool(sys.argv[4])
except ValueError:
    print("Error parsing command line arguments.")
    sys.exit(1)

model_path = "model_weights_1/config_15"
image_path = "dataset/processed/images/2.img_processed_norm.nii.gz"
output_path = "segmentation_2.nii.gz"
TH = 0.5

model = smp.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

subject = tio.Subject(
    image=tio.ScalarImage(Path(image_path)),
)
dataset = tio.SubjectsDataset([subject])
dataloader = tio.SubjectsLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False,
)
img_file = nib.load(Path(image_path))

with torch.no_grad():
    for batch in dataloader:
        img = batch["image"][tio.DATA].to(device)

        img_data = img.squeeze(1).permute(3, 0, 1, 2)

        logits = model(img_data)

        probs = torch.sigmoid(logits)
        segmentation = (probs > TH).long()

# reorder the segmentation back to normal
segmentation = segmentation.permute(1,2,3,0).squeeze().cpu()
# encode into new nifti image
segmented_img = nib.Nifti1Image(segmentation, affine=img_file.affine, header=img_file.header)
nib.save(segmented_img, output_path)

# visualise model output
if vis:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(segmentation, facecolors='r')
    plt.show()