import torch
import optuna
from torch.utils.data import DataLoader
from landcover.datasets import LandCoverDataset
from landcover.models.model import LandCoverModel
from landcover.utils import get_optimizer, get_loss_fn
from landcover.training.train import train, test
from landcover import DATA_PATH
from tqdm import tqdm
import wandb



class HyperparameterTuning:
    def __init__(self, n_trials, epochs, device):
        self.n_trials = n_trials
        self.epochs = epochs
        self.device = device
        self.train_dataset = LandCoverDataset((DATA_PATH / "dataset" / "clean" / "train"), pre_load=True)
        self.test_dataset = LandCoverDataset((DATA_PATH / "dataset" / "clean" / "test"), pre_load=True)

    def run(self):
        # perform optuna study
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=1,
            )
        )
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
        )

        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        wandb.init(
            project="land-cover-mapping",
            name=f"ResNet50/tuning/v0/best_trial",                                                                          # PLEASE CHANGE IF YOU WILL RUN IT AGAIN (the version and encoder name)
            config=best_params,
            notes=f"Best trial achieved test IoU of {best_value:.4f}",
            reinit=True
        )
        wandb.log({
            "best_test_iou": best_value,
        })
        wandb.finish()

        return study.trials_dataframe(), study.best_params

    def _objective(self, trial):
        # hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        encoder_name = trial.suggest_categorical("encoder_name", ["resnet34"])                                              # PLEASE CHANGE IF YOU WILL RUN IT AGAIN (the version)
        encoder_out_stride = trial.suggest_categorical("output_stride", [8, 16])
        encoder_depth = trial.suggest_categorical("encoder_depth", [4, 5])
        aspp_dropout = trial.suggest_float("aspp_dropout", 0.0, 0.5)
        decoder_channels = trial.suggest_categorical("decoder_channels", [64, 128, 256, 512])
        decoder_atrous_rates = trial.suggest_categorical(
            "decoder_atrous_rates",
            [
                (6, 12, 18),
                (12, 24, 36),
                (3, 6, 9),
                (12, 18, 24)
            ]
        )
        decoder_aspp_separable = trial.suggest_categorical("decoder_aspp_separable", [True, False])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        dice_weight = trial.suggest_float("dice_weight", 0.4, 0.8)

        # initialize wandb run
        wandb.init(
            project="land-cover-mapping",
            name=f"ResNet50/tuning/v0/trial-{trial.number}",                                                                # PLEASE CHANGE IF YOU WILL RUN IT AGAIN (the version)
            config={
                "lr": lr,
                "weight_decay": weight_decay,
                "encoder_name": encoder_name,
                "encoder_out_stride": encoder_out_stride,
                "encoder_depth": encoder_depth,
                "aspp_dropout": aspp_dropout,
                "decoder_atrous_rates": decoder_atrous_rates,
                "decoder_aspp_separable": decoder_aspp_separable,
                "decoder_channels": decoder_channels,
                "batch_size": batch_size,
                "dice_weight": dice_weight
            },
            dir=str(DATA_PATH),
            notes="This run displays the loss and iou graphs for the "
                  "training and testing sets during hyperparameter tuning.",
            reinit="finish_previous"
        )

        # dataloaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # model
        model = LandCoverModel(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=4,
            out_classes=9,
            encoder_output_stride=encoder_out_stride,
            encoder_depth=encoder_depth,
            decoder_aspp_dropout=aspp_dropout,
            decoder_atrous_rates=decoder_atrous_rates,
            decoder_aspp_separable=decoder_aspp_separable,
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
        for epoch in tqdm(range(self.epochs), desc=f"Trial {trial.number}", leave=False):
            avg_loss_train, avg_iou_train = train(model, train_loader, optimizer, loss_fn, device=self.device)
            avg_loss_test, avg_iou_test = test(model, test_loader, loss_fn, device=self.device)

            if not torch.isfinite(torch.tensor(avg_loss_train)) or not torch.isfinite(torch.tensor(avg_loss_test)):
                wandb.finish()
                raise optuna.exceptions.TrialPruned()

            if avg_iou_test > best_iou:
                best_iou = avg_iou_test

            wandb.log({
                "train_loss": avg_loss_train,
                "test_loss": avg_loss_test,
                "train_iou": avg_iou_train,
                "test_iou": avg_iou_test,
                "best_iou": best_iou,
                "epoch": epoch
            })

            trial.report(best_iou, epoch)
            if trial.should_prune():
                wandb.summary["pruned"] = True
                wandb.finish()
                raise optuna.exceptions.TrialPruned()

            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        wandb.summary["best_test_iou"] = best_iou
        wandb.finish()
        return best_iou