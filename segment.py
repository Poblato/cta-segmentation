import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
import nibabel as nib
from WeightedLoss import WeightedLoss

# import model params
# run model on specified file
# read params from command line
# 
model_path = "model_weights/0.pth"
image_path = "dataset/processed/images/1.img_processed_norm.nii.gz"
label_path = "dataset/processed/labels/1.label_processed_norm.nii.gz"
output_path = "segmentation_1.nii.gz"
TH = 0.5

model = smp.from_pretrained("model_weights/0.pth")

img = nib.load(image_path)
img_shape = img.header.get_data_shape()
print(img_shape)

lbl = nib.load(label_path)

assert(img_shape[0]==128 and img_shape[1]==128)

probs = model(img.get_fdata())
segmentation = (probs > TH).long()
lbl_data = lbl.get_fdata()

loss_fn = WeightedLoss(smp.losses.BINARY_MODE, dice_str=0, bce_str=0, focal_str=1, from_logits=True, 
            bce_pos_weight=torch.tensor([1.0]), focal_alpha=0.0019, focal_gamma=2)


acc = (lbl_data.detach().cpu().numpy() == segmentation.detach().cpu().numpy()).mean()
dice = loss_fn.compute_score(segmentation, lbl_data)
tp, fp, fn, tn = smp.metrics.get_stats(segmentation, (lbl_data > 0), mode='binary')
jaccard = smp.metrics.iou_score(tp, fp, fn, tn)
precision = smp.metrics.precision(tp, fp, fn, tn)
recall = smp.metrics.recall(tp, fp, fn, tn)

print(f"acc={acc},dice={dice},tp={tp},fp={fp},fn={fn},fp={fp},jaccard={jaccard},precision={precision},recall={recall}")

# encode into new nifti image
segmented_img = nib.Nifti1Image(segmentation, affine=img.affine, header=img.header)
nib.save(segmented_img, output_path)

# (optional) visualise model output
