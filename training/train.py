import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset import LandCoverDataset
from models.model import LandCoverModel
from utils.util import get_optimizer, get_loss_fn

DATA_DIR = 'data/tiles'
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 60
VAL_SPLIT = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = 9
IN_CHANNELS = 3


# Loss Functions
dice_loss_fn = smp.losses.DiceLoss(mode='multiclass')
ce_loss_fn = nn.CrossEntropyLoss()


def train(model_instance, data_loader, opt, loss_fn):
    model_instance.train()
    running_loss = 0
    running_iou = 0

    for images, masks in data_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        opt.zero_grad()

        outputs = model_instance(images)

        loss = loss_fn(outputs, masks)
        iou = metric_out(outputs, masks)

        loss.backward()
        opt.step()

        running_loss += loss.item()
        running_iou += iou

    avg_loss = running_loss / len(data_loader)
    avg_iou = running_iou / len(data_loader)

    return avg_loss, avg_iou

def test(model_instance, data_loader, loss_fn):
    model_instance.eval()
    running_loss = 0
    running_iou = 0
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model_instance(images)

            loss = loss_fn(outputs, masks)
            iou = metric_out(outputs, masks)

            running_loss += loss.item()
            running_iou += iou

    avg_loss = running_loss / len(data_loader)
    avg_iou = running_iou / len(data_loader)

    return avg_loss, avg_iou


def metric_out(outputs, masks):
    preds = torch.argmax(outputs, dim=1).long()
    masks.long()

    tp, fp, fn, tn = smp.metrics.get_stats(
        preds,
        masks,
        mode='multiclass',
        num_classes=NUM_CLASSES,
    )

    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro')

    return iou


if __name__ == '__main__':
    train_dataset = LandCoverDataset('data/train')
    test_dataset = LandCoverDataset('data/test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LandCoverModel(
        'efficientnet_b0',
        'imagenet',
        IN_CHANNELS,
        NUM_CLASSES,
        activation=None
    ).to(DEVICE)

    optimizer = get_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)

    loss_function = get_loss_fn()

    for epoch in range(EPOCHS):
        avg_loss_train, avg_iou_train = train(model, train_loader, optimizer, loss_function)
        avg_loss_test, avg_iou_test = test(model, test_loader, loss_function)