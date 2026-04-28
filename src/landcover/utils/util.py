import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import segmentation_models_pytorch as smp
from landcover import DATA_PATH
import numpy as np
import random

def get_loss_fn(dice_weight=0.5, ce_weight=0.5, device="cpu"):
    weights = np.load(DATA_PATH / "misc" / "class_weights.npy")
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
    ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=255)

    def loss(pred, target):
        return (dice_weight * dice_loss(pred, target)) + (ce_weight * ce_loss(pred, target))

    return loss

def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def compute_stats(outputs, masks, num_classes):
    preds = torch.argmax(outputs, dim=1)

    tp, fp, fn, tn = smp.metrics.get_stats(
        preds,
        masks,
        mode="multiclass",
        num_classes=num_classes,
        ignore_index=255
    )

    return tp, fp, fn, tn

def compute_metrics_from_stats(tp, fp, fn, tn, epsilon=1e-7):
    # Per-class metrics
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)

    # Macro averages
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()
    iou_macro = iou.mean()

    # Global accuracy
    accuracy = (tp.sum() + tn.sum()) / (
        tp.sum() + tn.sum() + fp.sum() + fn.sum() + epsilon
    )

    return {
        "mIoU": iou_macro.item(),
        "accuracy": accuracy.item(),
        "precision": precision_macro.item(),
        "recall": recall_macro.item(),
        "f1": f1_macro.item(),
        "per_class_iou": iou.detach().cpu().numpy(),
        "per_class_precision": precision.detach().cpu().numpy(),
        "per_class_recall": recall.detach().cpu().numpy(),
        "per_class_f1": f1.detach().cpu().numpy(),
    }

def crop(image, mask, y, x, p):
    return (
        image[:, y:y+p, x:x+p],
        mask[y:y+p, x:x+p],
    )

def pad_image(image, patch_size):
    *_, h, w = image.shape

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    padded = F.pad(image, (0, pad_w, 0, pad_h))

    return padded, pad_h, pad_w

def unpad_image(pred, pad_h, pad_w):
    h = pred.shape[-2]
    w = pred.shape[-1]
    return pred[..., :h - pad_h, :w - pad_w]

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)