import torch
import optuna
from torch.utils.data import DataLoader
from landcover.datasets import LandCoverDataset
from landcover.models.model import LandCoverModel
from landcover.utils import get_optimizer, get_loss_fn, seed_everything, seed_worker
from landcover.training.train import train
from landcover.evaluation.test import test
from landcover import DATA_PATH
from tqdm import tqdm
import wandb
from landcover import NUM_CLASSES


class HyperparameterTuning:
    def __init__(self, n_trials, epochs, encoder, version, device):
        self.n_trials = n_trials
        self.epochs = epochs
        self.encoder = encoder
        self.version = version
        self.device = device

    def run(self):
        # perform optuna study
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=10,
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
            name=f"{self.encoder}/tuning/{self.version}/best_trial",
            config=best_params,
            notes=f"Best trial achieved validation IoU of {best_value:.4f}",
            dir=str(DATA_PATH),
            reinit=True
        )
        wandb.log({
            "Best Validation IoU": best_value,
            "Best Tuning Trial": best_trial.number,
        })
        wandb.finish()

        return study.trials_dataframe(), study.best_params

    def _objective(self, trial):
        # apply seeding
        seed_everything(42 + trial.number)
        g = torch.Generator()
        g.manual_seed(42)

        # hyperparameters
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
        decoder_atrous_rates = trial.suggest_categorical("decoder_atrous_rates", [(6, 12, 18), (12, 18, 24)])
        decoder_aspp_separable = trial.suggest_categorical("decoder_aspp_separable", [False])
        decoder_channels = trial.suggest_categorical("decoder_channels", [64, 128, 256])
        encoder_depth = trial.suggest_categorical("encoder_depth", [4, 5])
        encoder_out_stride = trial.suggest_categorical("output_stride", [8, 16])
        patch_size = trial.suggest_categorical("patch_size", [256])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        aspp_dropout = trial.suggest_float("aspp_dropout", 0.2, 0.5)
        dice_weight = trial.suggest_float("dice_weight", 0.6, 0.8)

        # initialize wandb run
        wandb.init(
            project="land-cover-mapping",
            name=f"{self.encoder}/tuning/{self.version}/trial-{trial.number}",
            config={
                "Learning Rate": lr,
                "Weight Decay": weight_decay,
                "Encoder": self.encoder,
                "Encoder Output Stride": encoder_out_stride,
                "Encoder Depth": encoder_depth,
                "ASPP Dropout": aspp_dropout,
                "Decoder Atrous Rates": decoder_atrous_rates,
                "Decoder ASPP Separable": decoder_aspp_separable,
                "Decoder Channels": decoder_channels,
                "Batch Size": batch_size,
                "Dice Weight": dice_weight,
                "Patch Size": patch_size,
            },
            dir=str(DATA_PATH),
            reinit="finish_previous"
        )

        # dataloaders
        train_dataset = LandCoverDataset((DATA_PATH / "dataset" / "clean" / "train"), patch_size=patch_size, train_mode=True, pre_load=False)
        validation_dataset = LandCoverDataset((DATA_PATH / "dataset" / "clean" / "test"), patch_size=patch_size, train_mode=False, pre_load=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g, drop_last=False)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, worker_init_fn=seed_worker, generator=g, drop_last=False)

        # model
        model = LandCoverModel(
            encoder_name=self.encoder,
            encoder_weights="imagenet",
            in_channels=4,
            out_classes=NUM_CLASSES,
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # loss
        ce_weight = 1 - dice_weight
        loss_fn = get_loss_fn(dice_weight=dice_weight, ce_weight=ce_weight, device=self.device)

        # training loop
        best_m_iou = 0
        train_m_iou_at_best_val = 0
        for epoch in tqdm(range(self.epochs), desc=f"Trial {trial.number}", leave=False):
            train_metrics = train(model, train_loader, optimizer, loss_fn, device=self.device)
            val_metrics = test(model, validation_loader, loss_fn, patch_size, device=self.device)

            scheduler.step()

            if not torch.isfinite(torch.tensor(train_metrics["avg_loss"])) or not torch.isfinite(torch.tensor(val_metrics["avg_loss"])):
                wandb.finish()
                raise optuna.exceptions.TrialPruned()

            if val_metrics["mIoU"] > best_m_iou:
                best_m_iou = val_metrics["mIoU"]
                train_m_iou_at_best_val = train_metrics["mIoU"]

            wandb.log({
                "Train Loss": train_metrics["avg_loss"],
                "Validation Loss": val_metrics["avg_loss"],
                "Train mIoU": train_metrics["mIoU"],
                "Validation mIoU": val_metrics["mIoU"],
                "Best Validation mIoU": best_m_iou,
                "Train mIoU at Best Validation": train_m_iou_at_best_val,
                "Epoch": epoch
            })

            trial.report(val_metrics["mIoU"], epoch)
            if trial.should_prune():
                wandb.summary["Pruned"] = True
                wandb.finish()
                raise optuna.exceptions.TrialPruned()

            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        wandb.summary["Best Validation mIoU"] = best_m_iou
        wandb.summary["Train mIoU at Best Validation"] = train_m_iou_at_best_val
        wandb.finish()
        return best_m_iou