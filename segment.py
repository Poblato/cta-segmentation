import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
import nibabel as nib
import torchio as tio
from pathlib import Path
from WeightedLoss import WeightedLoss

# import model params
# run model on specified file
# read params from command line
# 
model_path = "model_weights/config_0"
image_path = "dataset/processed/images/1.img_processed_norm.nii.gz"
label_path = "dataset/processed/labels/1.label_processed_norm.nii.gz"
output_path = "segmentation_1.nii.gz"
TH = 0.5

model = smp.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

subject = tio.Subject(
    image=tio.ScalarImage(Path(image_path)),
    label=tio.LabelMap(Path(label_path))
)

dataset = tio.SubjectsDataset([subject])

dataloader = tio.SubjectsLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    shuffle=True,
)

with torch.no_grad():
    for batch in dataloader:
        img = batch["image"][tio.DATA].to(device)
        lbl = batch["label"][tio.DATA].to(device)

        img_data = img.squeeze(1).permute(3, 0, 1, 2)
        lbl_data = lbl.squeeze(1).permute(3, 0, 1, 2)

        logits = model(img_data)

        probs = torch.sigmoid(logits)
        segmentation = (probs > TH).long()

        acc = (lbl_layer == preds).float().mean().item()
        tp, fp, fn, tn = smp.metrics.get_stats(segmentation, (lbl_data > 0), mode='binary')
        tp = torch.sum(tp)
        fp = torch.sum(fp)
        fn = torch.sum(fn)
        tn = torch.sum(tn)
        dice = smp.metrics.f1_score(tp, fp, fn, tn)
        jaccard = smp.metrics.iou_score(tp, fp, fn, tn)
        precision = smp.metrics.precision(tp, fp, fn, tn)
        recall = smp.metrics.recall(tp, fp, fn, tn)

        print(f"acc={acc},dice={dice},tp={tp},fp={fp},fn={fn},tn={tn},jaccard={jaccard},precision={precision},recall={recall}")

# encode into new nifti image
segmented_img = nib.Nifti1Image(segmentation, affine=img.affine, header=img.header)
nib.save(segmented_img, output_path)

# (optional) visualise model output
