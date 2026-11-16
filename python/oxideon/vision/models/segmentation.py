"""U-Net semantic segmentation architecture in Oxideon Vision."""

from typing import List
from ...oxideon import Tensor as _Tensor
from ...nn import (
    Module,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    ConvTranspose2d,
    Sequential,
)


class DoubleConv(Module):
    """[Conv2d -> BatchNorm2d -> ReLU] x 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(),
        )

    def forward(self, x: _Tensor) -> _Tensor:
        return self.conv(x)


class UNet(Module):
    """Classic U-Net architecture for biomedical and generic image segmentation (Ronneberger et al., 2015)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, features: List[int] = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.downs = []
        self.ups = []
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling path)
        cur_in = in_channels
        for feature in features:
            block = DoubleConv(cur_in, feature)
            self.downs.append(block)
            setattr(self, f"down_{feature}", block)
            cur_in = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Upsampling path)
        for feature in reversed(features):
            up = ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            conv = DoubleConv(feature * 2, feature)
            self.ups.append((up, conv))
            setattr(self, f"up_trans_{feature}", up)
            setattr(self, f"up_conv_{feature}", conv)

        self.final_conv = Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: _Tensor) -> _Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx, (up, conv) in enumerate(self.ups):
            x = up(x)
            skip = skip_connections[idx]
            # Concatenate along channel dimension
            from ...oxideon import cat as _cat
            x = _cat([x, skip], dim=1)
            x = conv(x)

        return self.final_conv(x)


__all__ = [
    "UNet",
    "DoubleConv",
]
