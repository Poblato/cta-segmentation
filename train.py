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
from sklearn.model_selection import ParameterGrid
from WeightedLoss import WeightedLoss

train_list = []
valid_list = []
# test_list = []

files = os.listdir("dataset/processed/images")
# num_images = files.__len__()
num_images = 50

print("Loading dataset...")
for i in range(num_images):
    subject = tio.Subject(
        image=tio.ScalarImage(Path(f"dataset/processed/images/{i+1}.img_processed_norm.nii.gz")),
        label=tio.LabelMap(Path(f"dataset/processed/labels/{i+1}.label_processed_norm.nii.gz")),
    )
    if i < 0.8 * num_images:
        train_list.append(subject)
    else:
        valid_list.append(subject)

# init train, val, test sets
train_dataset = tio.SubjectsDataset(train_list)
valid_dataset = tio.SubjectsDataset(valid_list)
# test_dataset = tio.SubjectsDataset(test_list)

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
# print(f"Test size: {len(test_dataset)}")

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
# test_dataloader = tio.SubjectsLoader(
#     test_dataset,
#     batch_size=1,
#     num_workers=0,
#     shuffle=False,
# )



# Some training hyperparameters
EPOCHS = 10
T_MAX = EPOCHS * len(train_dataloader)
OUT_CLASSES = 1

def TrainModel(model, train_loader, val_loader, loss_fn, layer_batch_size):
    log = logging.getLogger('train')
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

        for batch in train_loader:
            img = batch["image"][tio.DATA].to(device)
            lbl = batch["label"][tio.DATA].to(device)

            for i in range(0, image_height, layer_batch_size):
                img_layer = img[:, :, :, :, i:i+layer_batch_size]
                lbl_layer = lbl[:, :, :, :, i:i+layer_batch_size]

                # reshape to something sensible (batch, channels, width, height)
                img_layer = img_layer.squeeze(1).permute(3, 0, 1, 2)
                lbl_layer = lbl_layer.squeeze(1).permute(3, 0, 1, 2)

                optimizer.zero_grad(set_to_none=True)

                logits = model(img_layer)
                T_loss = loss_fn(logits, lbl_layer)
                # print(T_loss)

                T_loss.backward()
                optimizer.step()

                running_loss += T_loss.item() * layer_batch_size
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs > TH).long()
                    running_acc += (lbl_layer.detach().cpu().numpy() == preds.detach().cpu().numpy()).mean() * layer_batch_size

        TrainLoss = running_loss / max(1, train_N*image_height)
        TrainAcc = running_acc / max(1, train_N*image_height)

        model.eval()
        running_loss = 0.0
        running_acc = 0.0

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
                    running_loss += V_loss.item() * layer_batch_size

                    probs = torch.sigmoid(logits)
                    preds = (probs > TH).long()
                    running_acc += (lbl_layer.detach().cpu().numpy() == preds.detach().cpu().numpy()).mean() * layer_batch_size

        ValLoss = running_loss / max(1, val_N*image_height)
        ValAcc = running_acc / max(1, val_N*image_height)

        # -------------- EMA + scheduler --------------
        ema_val = ValLoss if ema_val is None else alpha_ema*ValLoss + (1 - alpha_ema)*ema_val
        prev_lr = optimizer.param_groups[0]['lr']
        # scheduler.step(ema_val)  # schedule on EMA, not raw ValLoss
        new_lr = optimizer.param_groups[0]['lr']
        lr_drop = int(new_lr < prev_lr)

        scheduler.step()

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

def EvaluateModel(model, val_loader, loss_fn, layer_batch_size):
    val_N=len(val_loader.dataset)

    image_height = val_loader.dataset[0]["image"][tio.DATA].size(3)
    TH = 0.5

    log = logging.getLogger('eval')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    running_dice = 0.0

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
                running_loss += V_loss.item() * img.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs > TH).long()
                running_acc += (lbl_layer.detach().cpu().numpy() == preds.detach().cpu().numpy()).mean() * img.size(0)
                running_dice += smp.metrics.f1_score(preds, lbl_layer)

        ValLoss = running_loss / max(1, val_N)
        ValAcc = running_acc / max(1, val_N)
        DiceScore = running_dice / max(1, val_N)

        line=f"{ValLoss};{ValAcc};{DiceScore}"
        log.info(line)
        print(line)
        return {ValLoss, ValAcc, DiceScore}

# FIXME: Calculate positive prior probability for focal loss
pos_prior = 0.05
focal_strength = 2.0

# configure HPO
# params to vary: depth (def), layer_batch_size (def), loss fn (def), model encoder (maybe), LR (maybe)
grid = {
    'model_depth': [3, 4, 5],
    'layer_batch_size': [32, 64, 128],
    'dice_loss': [0, 1],
    'bce_loss': [0, 1],
    'focal_loss': [0, 1]
    # 'encoder': ['densenet121', 'resnet18'],
    # 'lr': [0.001, 0.01, 0.1]
}

param_grid = ParameterGrid(grid)
config_num = 0

logger = logging.getLogger("configs")
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("logs/hpo.log", mode="w"))

logger.info("ID, Depth, batch size, dice, bce, focal, val_loss, val_acc, dice_score")

for params in param_grid:

    print(params)
    # skip configs with 0 loss function
    if (params['dice_loss'] == 0 and params['bce_loss'] == 0 and params['focal_loss'] == 0):
        continue

    decoder_channels = [256, 128, 64, 32, 16]

    model = smp.Unet(
        encoder_name="densenet121",
        encoder_depth=params['model_depth'],
        encoder_weights=None,
        in_channels=1,
        decoder_channels=decoder_channels[0:params['model_depth']],
        classes=1
    )
    
    # apply equal weight to each enabled loss type
    dice_str = params['dice_loss']
    bce_str = params['bce_loss']
    focal_str = params['focal_loss']
    num = dice_str + bce_str + focal_str
    dice_str /= num
    bce_str /= num
    focal_str /= num

    # loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    # loss_fn = smp.losses.SoftBCEWithLogitsLoss(from_logits=True, pos_weight=torch.tensor([1.0]))
    # loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE, from_logits=True, alpha=pos_prior, gamma=focal_strength)
    loss_fn = WeightedLoss(smp.losses.BINARY_MODE, dice_str=dice_str, bce_str=bce_str, focal_str=focal_str, from_logits=True, 
                        bce_pos_weight=torch.tensor([1.0]), focal_alpha=pos_prior, focal_gamma=focal_strength)

    TrainModel(model, train_dataloader, valid_dataloader, loss_fn, params['layer_batch_size'])

    ValLoss, ValAcc, DiceScore = EvaluateModel(model, valid_dataloader, loss_fn, params['layer_batch_size'])

    # FIXME: log performance of each config
    line = f"{config_num},{params['model_depth']},{params['layer_batch_size']},{dice_str},{bce_str},{focal_str},{ValLoss},{ValAcc},{DiceScore}"
    logger.info(line)
    print(line)
    # Export model weights to file
    torch.save(model.state_dict(), f"model_weights/{config_num}.pth")
    config_num += 1

print("done")