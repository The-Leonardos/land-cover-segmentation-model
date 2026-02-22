import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp


def get_loss_fn(dice_weight=0.5, ce_weight=0.5):
    dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def loss(pred, target):
        return (dice_weight * dice_loss(pred, target)) + ce_weight * ce_loss(pred, target)

    return loss

def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
