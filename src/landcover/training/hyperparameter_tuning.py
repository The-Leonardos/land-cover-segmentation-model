import torch
import optuna
from torch.utils.data import DataLoader
from pathlib import Path
from src.landcover.datasets import LandCoverDataset
from src.landcover.models.model import LandCoverModel
from src.landcover.utils import get_optimizer, get_loss_fn
from src.landcover.training.train import train, test


class HyperparameterTuning:
    def __init__(self, n_trials, epochs, device):
        self.n_trials = n_trials
        self.epochs = epochs
        self.device = device
        self.train_dataset = LandCoverDataset('../data/train')
        self.test_dataset = LandCoverDataset('../data/test')
        pass

    def _objective(self, trial):
        # hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        encoder_name = trial.suggest_categorical('encoder_name', ['efficientnet_b0', 'resnet26', 'resnet34', 'resnet50'])
        encoder_out_stride = trial.suggest_categorical('output_stride', [8, 16])
        aspp_dropout = trial.suggest_float('aspp_dropout', 0.0, 0.5)
        decoder_channels = trial.suggest_categorical('decoder_channels', [64, 128, 256, 512])
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        dice_weight = trial.suggest_float('dice_weight', 0.4, 0.8)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        # model
        model = LandCoverModel(
            encoder_name,
            'imagenet',
            9,
            3,
            encoder_output_stride=encoder_out_stride,
            decoder_aspp_dropout=aspp_dropout,
            decoder_channels=decoder_channels,
            activation=None
        ).to(self.device)

        # optimizer
        optimizer = get_optimizer(model, lr=lr, weight_decay=weight_decay)

        # loss
        ce_weight = 1 - dice_weight
        loss_fn = get_loss_fn(dice_weight=dice_weight, ce_weight=ce_weight)

        # training loop
        best_iou = 0

        for epoch in range(self.epochs):
            train(model, train_loader, optimizer, loss_fn)
            _, avg_iou_test = test(model, test_loader, loss_fn)

            if avg_iou_test > best_iou:
                best_iou = avg_iou_test

            trial.report(best_iou, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        torch.cuda.empty_cache()

        return best_iou

    def run(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=self.n_trials)

        return study.trials_dataframe(), study.best_params_



