import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import segmentation_models_pytorch as smp
import numpy as np
from landcover import DATA_PATH


def get_loss_fn(dice_weight=0.5, ce_weight=0.5):
    weights = np.load(DATA_PATH / "misc" / "class_weights.npy")
    class_weights = torch.tensor(weights, dtype=torch.float32)
    class_weights = class_weights.cuda() if torch.cuda.is_available() else class_weights

    dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    def loss(pred, target):
        return (dice_weight * dice_loss(pred, target)) + ce_weight * ce_loss(pred, target)

    return loss

def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def compute_iou(outputs, masks):
    preds = torch.argmax(outputs, dim=1)

    tp, fp, fn, tn = smp.metrics.get_stats(
        preds,
        masks,
        mode='multiclass',
        num_classes=9,
        ignore_index=255
    )

    return tp, fp, fn, tn

def pad_image(image, patch_size):
    if (len(image.shape) == 4):
        _, _, h, w = image.shape
    else:
        _, h, w = image.shape

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    padded = F.pad(image, (0, pad_w, 0, pad_h))

    return padded, pad_h, pad_w

def unpad_image(pred, pad_h, pad_w):
    if pad_h > 0:
        pred = pred[:, :, :-pad_h, :]
    if pad_w > 0:
        pred = pred[:, :, :, :-pad_w]
    return pred