"""Deep Residual Learning for Image Recognition (ResNet) in Oxideon Vision."""

from typing import Any, Callable, List, Optional, Type, Union

from ...oxideon import Tensor as _Tensor
from ...nn import (
    Module,
    Linear,
    Conv2d,
    BatchNorm2d,
    ReLU,
    Sequential,
    AdaptiveAvgPool2d,
)


class BasicBlock(Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Module] = None,
    ):
        super().__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(planes)
        self.relu = ReLU()
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: _Tensor) -> _Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(Module):
    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Module] = None,
    ):
        super().__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: _Tensor) -> _Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet(Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
    ):
        super().__init__()
        self.in_planes = 64

        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return Sequential(*layers)

    def forward(self, x: _Tensor) -> _Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def resnet18(num_classes: int = 1000) -> ResNet:
    """ResNet-18 model architecture."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes: int = 1000) -> ResNet:
    """ResNet-34 model architecture."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes: int = 1000) -> ResNet:
    """ResNet-50 model architecture."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


__all__ = [
    "ResNet",
    "BasicBlock",
    "Bottleneck",
    "resnet18",
    "resnet34",
    "resnet50",
]
