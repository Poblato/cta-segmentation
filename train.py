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
num_images = files.__len__()
# num_images = 50

print("Loading dataset...")
for i in range(num_images):
    subject = tio.Subject(
        image=tio.ScalarImage(Path(f"dataset/processed/images/{i+1}.img_processed_norm.nii.gz")),
        label=tio.LabelMap(Path(f"dataset/processed/labels/{i+1}.label_processed_norm.nii.gz"))
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
EPOCHS = 20
OUT_CLASSES = 1

def TrainModel(model, train_loader, val_loader, loss_fn, lr, layer_batch_size):
    log = logging.getLogger('train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    TH = 0.5

    ES_PATIENCE = 5

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min=1e-5)

    # --- best trackers ---
    best_val_loss = np.inf
    best_epoch    = -1
    best_lr_at_best = lr
    accuracy_at_best = np.nan

    no_improve = 0

    val_N=len(val_loader.dataset)
    train_N=len(train_loader.dataset)

    image_height = train_loader.dataset[0]["image"][tio.DATA].size(3)

    print("Training model...")
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
                    running_acc += (lbl_layer == preds).float().mean().item() * layer_batch_size

        ValLoss = running_loss / max(1, val_N*image_height)
        ValAcc = running_acc / max(1, val_N*image_height)

        prev_lr = optimizer.param_groups[0]['lr']
        new_lr = optimizer.param_groups[0]['lr']
        lr_drop = int(new_lr < prev_lr)

        scheduler.step()

        # -------------- early stopping test -----------
        improved = ValLoss < best_val_loss
        if improved:
            best_val_loss = ValLoss
            best_epoch = epoch
            accuracy_at_best = ValAcc
            no_improve = 0
        else:
            no_improve += 1

        es_triggered = int(no_improve >= ES_PATIENCE)
        line=f"{epoch};{TrainLoss};{TrainAcc};{ValLoss};{ValAcc};{new_lr};{lr_drop};{best_val_loss};{accuracy_at_best};{best_epoch}"
        log.info(line)
        print(line)
        if es_triggered: 
            print("ES triggered")
            break
    # Delete optimiser and scheduler to free gpu memory
    del optimizer
    del scheduler

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
    tp, fp, fn, tn = 0, 0, 0, 0

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
                running_acc += (lbl_layer == preds).float().mean().item() * layer_batch_size
                tp_t, fp_t, fn_t, tn_t = smp.metrics.get_stats(preds, (lbl_layer > 0), mode='binary')
                tp += torch.sum(tp_t)
                fp += torch.sum(fp_t)
                fn += torch.sum(fn_t)
                tn += torch.sum(tn_t)

        Loss = running_loss / max(1, val_N*image_height)
        Acc = running_acc / max(1, val_N*image_height)
        DiceScore = smp.metrics.f1_score(tp, fp, fn, tn)
        JaccardIndex = smp.metrics.iou_score(tp, fp, fn, tn)
        Precision = smp.metrics.precision(tp, fp, fn, tn)
        Recall = smp.metrics.recall(tp, fp, fn, tn)

        line=f"{Loss};{Acc};{DiceScore};{JaccardIndex};{Precision};{Recall}"
        log.info(line)
        print(line)
        return {Loss, Acc, DiceScore, JaccardIndex, Precision, Recall}

focal_strength = 2.0

# configure HPO
# grid = {
#     # 'model_depth': [3, 4],
#     # 'layer_batch_size': [16, 32, 64],
#     'loss': ['Dice', 'BCE', 'Focal'],
#     'alpha' : [0.6, 0.7, 0.8],
#     'gamma' : [1, 2, 3],
#     'lr' : [1e-2, 1e-3, 1e-4]
# }
grid = {
    'dice': [0, 0.2, 0.4, 0.6, 0.8, 1],
    'bce': [0, 0.2, 0.4, 0.6, 0.8, 1]
}

param_grid = ParameterGrid(grid)
config_num = 0

logger = logging.getLogger("configs")
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("logs/loss.log", mode="a"))

train_log = logging.getLogger("train")
train_log.setLevel(logging.INFO)
train_log.addHandler(logging.FileHandler("logs/train.log", mode="a"))

# logger.info("ID, dice, bce, focal, loss, acc, dice_score, jaccard index, precision, recall")
logger.info("ID, Depth, Loss, Batch, LR, Alpha, Gamma, loss, acc, dice_score, jaccard index, precision, recall")

train_log.info("Epoch, TrainLoss, TrainAcc, ValLoss, ValAcc, newLR, LRdrop, bestValLoss, AccAtBest, BestEpoch")

for params in param_grid:
    print(params)
    # skip configs with invalid loss function
    if (params['dice'] + params['bce'] > 1):
        continue

    # dont bother with focal param optimisation is loss fucntion isnt focal
    # if (params['loss'] != "Focal" and (params['alpha'] != 0.6 or params['gamma'] != 1)):
    #     continue
    
    # if config_num < 9:
    #     config_num += 1
    #     continue
    
    decoder_channels = [256, 128, 64, 32, 16]

    model_depth = 4
    layer_batch_size = 16
    alpha = 0.7
    gamma = 2
    lr = 1e-3

    # model_depth = params['model_depth']
    # layer_batch_size = params['layer_batch_size']
    # lr = params['lr']
    # alpha = params['alpha']
    # gamma = params['gamma']

    model = smp.Unet(
        encoder_name="densenet121",
        encoder_depth=model_depth,
        encoder_weights=None,
        in_channels=1,
        decoder_channels=decoder_channels[0:model_depth],
        classes=1
    )
    
    # apply equal weight to each enabled loss type
    dice_str = params['dice']
    bce_str = params['bce']
    focal_str = max(0, 1 - dice_str - bce_str)
    # dice_str = params['loss'] == 'Dice'
    # bce_str = params['loss'] == 'BCE'
    # focal_str = params['loss'] == 'Focal'

    loss_fn = WeightedLoss(smp.losses.BINARY_MODE, dice_str=dice_str, bce_str=bce_str, focal_str=focal_str, from_logits=True, 
                        bce_pos_weight=torch.tensor([1.0], device=("cuda" if torch.cuda.is_available() else "cpu")), focal_alpha=alpha, focal_gamma=gamma)

    TrainModel(model, train_dataloader, valid_dataloader, loss_fn, lr, layer_batch_size)

    Loss, Acc, DiceScore, JaccardIndex, Precision, Recall = EvaluateModel(model, valid_dataloader, loss_fn, layer_batch_size)

    # Log performance of each config
    line = f"{config_num},{dice_str},{bce_str},{focal_str},{Loss},{Acc},{DiceScore},{JaccardIndex},{Precision},{Recall}"
    # line = f"{config_num},{model_depth},{params['loss']},{layer_batch_size},{lr},{alpha},{gamma},{Loss},{Acc},{DiceScore},{JaccardIndex},{Precision},{Recall}"
    logger.info(line)
    print(line)
    # Export model weights to file
    model.cpu()
    model.save_pretrained(f"model_weights/config_{config_num}")
    config_num += 1

    # clear model to free GPU memory
    del loss_fn
    del model
    torch.cuda.empty_cache()

print("done")