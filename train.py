import os
import torch
# import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
# from torch.utils.data import DataLoader
from pathlib import Path
import torchio as tio
import logging
import numpy as np

train_list = []
valid_list = []
test_list = []

files = os.listdir("dataset/processed/images")
# num_images = files.__len__()
num_images = 50

print("Loading dataset...")
for i in range(num_images):
    subject = tio.Subject(
        image=tio.ScalarImage(Path(f"dataset/processed/images/{i+1}.img_processed_norm.nii.gz")),
        label=tio.LabelMap(Path(f"dataset/processed/labels/{i+1}.label_processed_norm.nii.gz")),
    )
    if i < 0.6 * num_images:
        train_list.append(subject)
    elif i < 0.8 * num_images:
        valid_list.append(subject)
    else:
        test_list.append(subject)

# init train, val, test sets
train_dataset = tio.SubjectsDataset(train_list)
valid_dataset = tio.SubjectsDataset(valid_list)
test_dataset = tio.SubjectsDataset(test_list)

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()

train_dataloader = tio.SubjectsLoader(
    train_dataset,
    batch_size=1,
    num_workers=0,
    shuffle=True,
)
valid_dataloader = tio.SubjectsLoader(
    valid_dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False,
)
test_dataloader = tio.SubjectsLoader(
    test_dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False,
)

# Some training hyperparameters
EPOCHS = 10
T_MAX = EPOCHS * len(train_dataloader)
OUT_CLASSES = 1

# FIXME: read batch size from image height
layer_batch_size = 128

def TrainModel(model, train_loader, val_loader):
    log = logging.getLogger('INNER_train')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    LR = 2e-4
    TH = 0.5

    sch_patience, sch_cooldown = 2, 1
    rel_thresh = 5e-3
    ES_PATIENCE = sch_patience + sch_cooldown + 2
    alpha_ema = 0.30  # smoothing for EMA of val loss

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
    # criterion = nn.BCEWithLogitsLoss()
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    # --- best trackers ---
    best_val_loss = np.inf
    best_epoch    = -1
    best_lr_at_best = LR
    accuracy_at_best = np.nan

    # --- EMA & patience ---
    ema_val = None
    best_ema = np.inf
    no_improve = 0

    val_N=len(val_loader.dataset)
    train_N=len(train_loader.dataset)

    image_height = train_loader.dataset[0]["image"][tio.DATA].size(3)

    print("↳ Training model...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        y_true, y_pred = [], []

        # need to input layer by layer rather than as whole images

        for batch in train_loader:
            img = batch["image"][tio.DATA].to(device)
            lbl = batch["label"][tio.DATA].to(device)

            for i in range(0, image_height, layer_batch_size):
                img_layer = img[:, :, :, :, i:i+layer_batch_size]
                lbl_layer = lbl[:, :, :, :, i:i+layer_batch_size]

                # reshape to something sensible (batch, channels, width, height)
                img_layer = img_layer.squeeze(1).permute(3, 0, 1, 2)
                lbl_layer = lbl_layer.squeeze(1).permute(3, 0, 1, 2)

                # optimizer.zero_grad(set_to_none=True)

                logits = model(img_layer)
                T_loss = loss_fn(logits, lbl_layer)

                T_loss.backward()
                optimizer.step()

                running_loss += T_loss.item() * img.size(0)
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs > TH).long()
                    running_acc += (lbl_layer.detach().cpu().numpy() == preds.detach().cpu().numpy()).mean() * img.size(0)
                    # y_true.append(lbl_layer.detach().cpu().numpy())
                    # y_pred.append(preds.detach().cpu().numpy())

        TrainLoss = running_loss / max(1, train_N)
        # y_true = np.concatenate(y_true).reshape(-1)
        # y_pred = np.concatenate(y_pred).reshape(-1)
        # TrainAcc  = (y_true == y_pred).mean()
        TrainAcc = running_acc / max(1, train_N)

        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        y_true, y_prob, y_pred = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"][tio.DATA].to(device)
                lbl = batch["label"][tio.DATA].to(device)

                for i in range(0, image_height, layer_batch_size):
                    img_layer = img[:, :, :, :, i:i+layer_batch_size]
                    lbl_layer = lbl[:, :, :, :, i:i+layer_batch_size]

                    # reshape to something sensible (batch, channels, width, height)
                    img_layer = img_layer.squeeze(1).permute(3, 0, 1, 2)
                    lbl_layer = lbl_layer.squeeze(1).permute(3, 0, 1, 2)

                    logits = model(img_layer)
                    V_loss = loss_fn(logits, lbl_layer)
                    running_loss += V_loss.item() * lbl_layer.size(0)

                    probs = torch.sigmoid(logits)
                    preds = (probs > TH).long()
                    running_acc += (lbl_layer.detach().cpu().numpy() == preds.detach().cpu().numpy()).mean() * img.size(0)

                    # y_true.append(lbl_layer.detach().cpu().numpy())
                    # y_prob.append(probs.detach().cpu().numpy())
                    # y_pred.append(preds.detach().cpu().numpy())

        ValLoss = running_loss / max(1, val_N)

        # y_true  = np.concatenate(y_true).reshape(-1)
        # y_prob  = np.concatenate(y_prob).reshape(-1)
        # y_pred  = np.concatenate(y_pred).reshape(-1)

        ValAcc = running_acc / max(1, val_N)

        # ValAcc  = (y_true == y_pred).mean()
        # AUC = roc_auc_score(y_true, y_prob)

        # -------------- EMA + scheduler --------------
        ema_val = ValLoss if ema_val is None else alpha_ema*ValLoss + (1 - alpha_ema)*ema_val
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(ema_val)  # schedule on EMA, not raw ValLoss
        new_lr = optimizer.param_groups[0]['lr']
        lr_drop = int(new_lr < prev_lr)

        # -------------- early stopping test -----------
        improved = ema_val < best_ema * (1 - rel_thresh)
        if improved:
            best_ema = ema_val
            best_val_loss = ValLoss
            best_epoch = epoch
            best_lr_at_best = new_lr
            accuracy_at_best = ValAcc
            no_improve = 0
        else:
            no_improve += 1

        # es_triggered = int(no_improve >= ES_PATIENCE)
        line=f"{epoch};{TrainLoss};{TrainAcc};{ValLoss};{ValAcc};{ema_val};{new_lr};{lr_drop};{best_val_loss};{best_epoch}"
        log.info(line)
        print(line)
        # if es_triggered: break


model = smp.Unet(
    encoder_name="densenet121",
    encoder_weights=None,
    in_channels=1,
    classes=1
)

TrainModel(model, train_dataloader, valid_dataloader)

print("done?")

# FIXME: run validation and test datasets for final evaluation


# trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)

# trainer.fit(
#     model,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=valid_dataloader,
# )

# run validation dataset
# valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
# print(valid_metrics)

# # run test dataset
# test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
# print(test_metrics)

