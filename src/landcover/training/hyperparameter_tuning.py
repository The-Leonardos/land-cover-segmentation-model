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
        lr = trial.suggest_float("lr", 5e-5, 2e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 4e-4, log=True)
        decoder_atrous_rates = trial.suggest_categorical("decoder_atrous_rates", [(12, 18, 24)])
        decoder_aspp_separable = trial.suggest_categorical("decoder_aspp_separable", [False])
        decoder_channels = trial.suggest_categorical("decoder_channels", [128, 256])
        encoder_depth = trial.suggest_categorical("encoder_depth", [5])
        encoder_out_stride = trial.suggest_categorical("output_stride", [16])
        patch_size = trial.suggest_categorical("patch_size", [256])
        batch_size = trial.suggest_categorical("batch_size", [16])
        aspp_dropout = trial.suggest_float("aspp_dropout", 0.2, 0.4)
        dice_weight = trial.suggest_float("dice_weight", 0.4, 0.6)

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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, worker_init_fn=seed_worker, generator=g)

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
        loss_fn = get_loss_fn(dice_weight=dice_weight, ce_weight=ce_weight)

        # training loop
        best_m_iou = 0
        train_m_iou_at_best_val = 0
        for epoch in tqdm(range(self.epochs), desc=f"Trial {trial.number}", leave=False):
            avg_train_loss, train_m_iou = train(model, train_loader, optimizer, loss_fn, device=self.device)
            avg_val_loss, val_m_iou = test(model, validation_loader, loss_fn, patch_size, device=self.device)

            scheduler.step()

            if not torch.isfinite(torch.tensor(avg_train_loss)) or not torch.isfinite(torch.tensor(avg_val_loss)):
                wandb.finish()
                raise optuna.exceptions.TrialPruned()

            if val_m_iou > best_m_iou:
                best_m_iou = val_m_iou
                train_m_iou_at_best_val = train_m_iou

            wandb.log({
                "Train Loss": avg_train_loss,
                "Validation Loss": avg_val_loss,
                "Train mIoU": train_m_iou,
                "Validation mIoU": val_m_iou,
                "Best Validation mIoU": best_m_iou,
                "Train mIoU at Best Validation": train_m_iou_at_best_val,
                "Epoch": epoch
            })

            trial.report(val_m_iou, epoch)
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