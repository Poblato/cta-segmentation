import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
import torchio as tio
from pathlib import Path
from WeightedLoss import WeightedLoss
import logging

TH = 0.5

logger = logging.getLogger('eval')
logger.addHandler(logging.FileHandler("logs/eval_hpo.log", mode='w'))
logger.setLevel(logging.INFO)
logger.info("Config Num, Acc, TP, FP, FN, TN, Dice, Jaccard, Precision, Recall")

images = np.arange(1, 201)
num_images = 200
data_list = []
for j in images:
    subject = tio.Subject(
        image=tio.ScalarImage(Path(f"dataset/processed/images/{j}.img_processed_norm.nii.gz")),
        label=tio.LabelMap(Path(f"dataset/processed/labels/{j}.label_processed_norm.nii.gz"))
    )
    data_list.append(subject)
dataset = tio.SubjectsDataset(data_list)
dataloader = tio.SubjectsLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False,
)

num_models = 33
for i in range(num_models):
    model = smp.from_pretrained(Path(f"model_weights_hpo/config_{i}"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tp, fp, fn, tn, acc = 0, 0, 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch["image"][tio.DATA].to(device)
            lbl = batch["label"][tio.DATA].to(device)

            img_data = img.squeeze(1).permute(3, 0, 1, 2)
            lbl_data = lbl.squeeze(1).permute(3, 0, 1, 2)

            logits = model(img_data)

            probs = torch.sigmoid(logits)
            segmentation = (probs > TH).long()

            acc += (segmentation == lbl_data).float().mean().item()
            tp_t, fp_t, fn_t, tn_t = smp.metrics.get_stats(segmentation, (lbl_data > 0), mode='binary')
            tp += torch.sum(tp_t)
            fp += torch.sum(fp_t)
            fn += torch.sum(fn_t)
            tn += torch.sum(tn_t)

    acc /= num_images
    dice = smp.metrics.f1_score(tp, fp, fn, tn)
    jaccard = smp.metrics.iou_score(tp, fp, fn, tn)
    precision = smp.metrics.precision(tp, fp, fn, tn)
    recall = smp.metrics.recall(tp, fp, fn, tn)
    line = f"{i},{acc},{tp},{fp},{fn},{tn},{dice},{jaccard},{precision},{recall}" 
    print(line)
    logger.info(line)
