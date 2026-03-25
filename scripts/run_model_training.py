import torch
from torch.utils.data import DataLoader
from landcover.datasets import LandCoverDataset
from landcover.models import LandCoverModel
from landcover.utils import get_optimizer, get_loss_fn
from landcover.training.train import train
from landcover.evaluation.test import test
from landcover import DATA_PATH
import datetime
import wandb
from dotenv import load_dotenv
from tqdm import tqdm
import os


if __name__ == '__main__':
    # load credentials
    load_dotenv("../config.env")
    KEY = os.getenv("WANDB_API_KEY")

    # auth login to weights and biases
    wandb.login(key=KEY)

    # check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device]: {device.upper()}")
    torch.backends.cudnn.benchmark = True

    # hyperparameters
    lr = 0.003920488235974936
    weight_decay = 0.00009615552966434236
    encoder_name = "resnet50"
    encoder_out_stride = 8
    encoder_depth = 4
    aspp_dropout = 0.3615941301989724
    decoder_channels = 256
    decoder_atrous_rates = (12, 24, 36)
    decoder_aspp_separable = False
    batch_size = 32
    dice_weight = 0.5344173714762183
    patch_size = 512

    # training setup
    epochs = 220
    model_path = DATA_PATH / "models" / encoder_name
    model_path.mkdir(parents=True, exist_ok=True)

    # time of training
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # initialize wandb run
    wandb.init(
        project="land-cover-mapping",
        name=f"{encoder_name}/training/{date_time}",
        config={
            "Learning Rate": lr,
            "Weight Decay": weight_decay,
            "Encoder": encoder_name,
            "Encoder Output Stride": encoder_out_stride,
            "Encoder Depth": encoder_depth,
            "ASPP Dropout": aspp_dropout,
            "Decoder Atrous Rates": decoder_atrous_rates,
            "Decoder ASPP Separable": decoder_aspp_separable,
            "Decoder Channels": decoder_channels,
            "Batch Size": batch_size,
            "Dice Weight": dice_weight,
            "Patch Size": patch_size
        },
        dir=str(DATA_PATH)
    )

    # dataloaders
    train_dataset = LandCoverDataset((DATA_PATH / "dataset" / "clean" / "train"), train_mode=True, pre_load=True)
    validation_dataset = LandCoverDataset((DATA_PATH / "dataset" / "clean" / "test"), train_mode=False, pre_load=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

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
    ).to(device)

    # optimizer
    optimizer = get_optimizer(model, lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # loss
    ce_weight = 1 - dice_weight
    loss_fn = get_loss_fn(dice_weight=dice_weight, ce_weight=ce_weight)

    # training loop
    best_iou = 0
    best_epoch = 0
    for epoch in tqdm(range(epochs), desc=f"{encoder_name} Training", leave=False):
        avg_train_loss, train_m_iou = train(model, train_loader, optimizer, loss_fn, device=device)
        avg_val_loss, val_m_iou = test(model, validation_loader, loss_fn, patch_size, device=device)

        if not torch.isfinite(torch.tensor(avg_train_loss)) or not torch.isfinite(torch.tensor(avg_val_loss)):
            print("Loss became NaN. Stopping training")
            wandb.summary["Early Stop"] = "Training stopped due to NaN values"
            wandb.summary["NaN Train Loss"] = avg_train_loss
            wandb.summary["NaN Validation Loss"] = avg_val_loss
            wandb.finish()
            break

        if val_m_iou > best_iou:
            best_iou = val_m_iou
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                model_path / f"deeplabv3plus_{encoder_name}_best_{epoch}.pth"
            )

        scheduler.step()

        wandb.log({
            "Train Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
            "Train IoU": train_m_iou,
            "Validation IoU": val_m_iou,
            "best IoU": best_iou,
            "Best Epoch": best_epoch,
            "learning rate": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        })

        torch.cuda.empty_cache()

    wandb.summary["Best Validation IoU"] = best_iou
    wandb.summary["Best Epoch"] = best_epoch
    wandb.finish()

    # save final epoch model
    torch.save(
        model.state_dict(),
        model_path / f"deeplabv3plus_{encoder_name}_best_{epochs}.pth"
    )