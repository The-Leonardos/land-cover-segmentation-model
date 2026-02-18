import torch.nn as nn
import segmentation_models_pytorch as smp

class LandCoverModel(nn.Module):
    def __init__(self, encoder_name, encoder_weights, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs
        )

    def forward(self, x):
        return self.model(x)