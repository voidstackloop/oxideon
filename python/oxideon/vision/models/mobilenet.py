"""MobileNetV3 efficient mobile architecture in Oxideon Vision."""

from typing import List, Optional

from ...oxideon import Tensor as _Tensor
from ...nn import (
    Module,
    Linear,
    Conv2d,
    BatchNorm2d,
    Sequential,
    AdaptiveAvgPool2d,
    Dropout,
    SiLU,
    ReLU,
    Sigmoid,
)


class SqueezeExcitation(Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, in_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = max(1, in_channels // squeeze_factor)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc1 = Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.relu = ReLU()
        self.fc2 = Conv2d(squeeze_channels, in_channels, kernel_size=1)
        self.sigmoid = Sigmoid()

    def forward(self, x: _Tensor) -> _Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class InvertedResidual(Module):
    """Inverted Residual block with optional Squeeze-and-Excitation and depthwise separable convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        use_se: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))
        layers: List[Module] = []

        if expand_ratio != 1:
            layers.extend([
                Conv2d(in_channels, hidden_dim, kernel_size=1),
                BatchNorm2d(hidden_dim),
                SiLU(),
            ])

        # Depthwise
        layers.extend([
            Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            BatchNorm2d(hidden_dim),
            SiLU(),
        ])

        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))

        # Project
        layers.extend([
            Conv2d(hidden_dim, out_channels, kernel_size=1),
            BatchNorm2d(out_channels),
        ])

        self.block = Sequential(*layers)

    def forward(self, x: _Tensor) -> _Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class MobileNetV3(Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # Stem
        self.stem = Sequential(
            Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(16),
            SiLU(),
        )

        # Inverted residual blocks
        self.features = Sequential(
            InvertedResidual(16, 16, kernel_size=3, stride=2, expand_ratio=1.0, use_se=True),
            InvertedResidual(16, 24, kernel_size=3, stride=2, expand_ratio=4.0, use_se=False),
            InvertedResidual(24, 24, kernel_size=3, stride=1, expand_ratio=3.0, use_se=False),
            InvertedResidual(24, 40, kernel_size=5, stride=2, expand_ratio=3.0, use_se=True),
            InvertedResidual(40, 40, kernel_size=5, stride=1, expand_ratio=3.0, use_se=True),
            InvertedResidual(40, 96, kernel_size=5, stride=1, expand_ratio=6.0, use_se=True),
        )

        self.pool = AdaptiveAvgPool2d((1, 1))
        self.classifier = Sequential(
            Linear(96, 256),
            SiLU(),
            Dropout(0.2),
            Linear(256, num_classes),
        )

    def forward(self, x: _Tensor) -> _Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def mobilenet_v3_small(num_classes: int = 1000) -> MobileNetV3:
    """Constructs MobileNetV3-Small architecture."""
    return MobileNetV3(num_classes=num_classes)


__all__ = [
    "MobileNetV3",
    "InvertedResidual",
    "SqueezeExcitation",
    "mobilenet_v3_small",
]
