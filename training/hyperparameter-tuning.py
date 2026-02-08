import torch
import optuna
from torch.utils.data import DataLoader
from pathlib import Path
from datasets.dataset import LandCoverDataset
from models.model import LandCoverModel
from utils.util import get_optimizer, get_loss_fn
from training.train import train, test


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 9
IN_CHANNELS = 3
EPOCHS = 30


def objective(trial):
    # hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    encoder_name = trial.suggest_categorical('encoder_name', ['efficientnet_b0', 'resnet26', 'resnet34', 'resnet50'])
    encoder_out_stride = trial.suggest_categorical('output_stride', [8, 16])
    aspp_dropout = trial.suggest_float('aspp_dropout', 0.0, 0.5)
    decoder_channels = trial.suggest_categorical('decoder_channels', [64, 128, 256, 512])
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    dice_weight = trial.suggest_float('dice_weight', 0.4, 0.8)

    # dataset and dataloaders
    train_dataset = LandCoverDataset('../data/train')
    test_dataset = LandCoverDataset('../data/test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    model = LandCoverModel(
        encoder_name,
        'imagenet',
        IN_CHANNELS,
        NUM_CLASSES,
        encoder_output_stride=encoder_out_stride,
        decoder_aspp_dropout=aspp_dropout,
        decoder_channels=decoder_channels,
        activation=None
    ).to(DEVICE)

    # optimizer
    optimizer = get_optimizer(model, lr=lr, weight_decay=weight_decay)

    # loss
    ce_weight = 1 - dice_weight
    loss_fn = get_loss_fn(dice_weight=dice_weight, ce_weight=ce_weight)

    # training loop
    best_iou = 0

    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer, loss_fn)
        _, avg_iou_test = test(model, test_loader, loss_fn)

        if avg_iou_test > best_iou:
            best_iou = avg_iou_test

        trial.report(best_iou, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_iou

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    df = study.trials_dataframe()

    csv_path = Path('../data/hyperparameter_tuning/deeplabv3plus_tuning_results.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)