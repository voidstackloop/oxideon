"""Computer vision transforms and data augmentation pipeline for Oxideon."""

from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


class Compose:
    """Composes several transforms together in sequence."""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class ToTensor:
    """Converts a PIL Image, NumPy array, or sequence into an Oxideon `Tensor`.

    Scales pixel values in [0, 255] to [0.0, 1.0] and transposes (H, W, C) to (C, H, W).
    """

    def __call__(self, pic) -> _Tensor:
        if isinstance(pic, _Tensor):
            return pic

        arr = np.asarray(pic, dtype=np.float32)
        if arr.size == 0:
            return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

        if arr.ndim == 2:  # Grayscale [H, W] -> [1, H, W]
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):  # [H, W, C] -> [C, H, W]
            arr = np.transpose(arr, (2, 0, 1))

        # Normalize [0, 255] -> [0.0, 1.0] if integer range
        if np.max(arr) > 1.0:
            arr = arr / 255.0

        return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Normalize:
    """Normalizes a float32 tensor image with mean and standard deviation:
    `output[channel] = (input[channel] - mean[channel]) / std[channel]`
    """

    def __init__(self, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]]):
        self.mean = [mean] if isinstance(mean, (int, float)) else list(mean)
        self.std = [std] if isinstance(std, (int, float)) else list(std)

    def __call__(self, tensor: _Tensor) -> _Tensor:
        arr = tensor.to_numpy() if hasattr(tensor, "to_numpy") else np.asarray(tensor)
        if arr.ndim == 4:
            mean_arr = np.array(self.mean, dtype=np.float32).reshape(1, -1, 1, 1)
            std_arr = np.array(self.std, dtype=np.float32).reshape(1, -1, 1, 1)
        elif arr.ndim == 3:
            mean_arr = np.array(self.mean, dtype=np.float32).reshape(-1, 1, 1)
            std_arr = np.array(self.std, dtype=np.float32).reshape(-1, 1, 1)
        elif arr.ndim == 2:
            mean_arr = np.array(self.mean, dtype=np.float32).reshape(-1, 1)
            std_arr = np.array(self.std, dtype=np.float32).reshape(-1, 1)
        else:
            mean_arr = np.array(self.mean, dtype=np.float32)
            std_arr = np.array(self.std, dtype=np.float32)

        if getattr(tensor, "requires_grad", False):
            mean_t = _from_numpy(np.ascontiguousarray(mean_arr))
            std_t = _from_numpy(np.ascontiguousarray(std_arr))
            return (tensor - mean_t) / std_t

        normed = (arr - mean_arr) / np.maximum(std_arr, 1e-12)
        return _from_numpy(np.ascontiguousarray(normed, dtype=np.float32))

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class RandomHorizontalFlip:
    """Horizontally flips the given image randomly with a given probability."""

    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        self.p = p
        self.seed = seed

    def __call__(self, img):
        rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
        if rng.rand() < self.p:
            if isinstance(img, _Tensor):
                arr = img.to_numpy()
                flipped = np.ascontiguousarray(arr[..., ::-1])
                return _from_numpy(flipped)
            arr = np.asarray(img)
            return arr[..., ::-1]
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip:
    """Vertically flips the given image randomly with a given probability."""

    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        self.p = p
        self.seed = seed

    def __call__(self, img):
        rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
        if rng.rand() < self.p:
            if isinstance(img, _Tensor):
                arr = img.to_numpy()
                flipped = np.ascontiguousarray(arr[..., ::-1, :])
                return _from_numpy(flipped)
            arr = np.asarray(img)
            return arr[..., ::-1, :]
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class CenterCrop:
    """Crops the given image at the center to the target size."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, img):
        arr = img.to_numpy() if isinstance(img, _Tensor) else np.asarray(img)
        th, tw = self.size
        h, w = arr.shape[-2], arr.shape[-1]

        i = max(0, int(round((h - th) / 2.0)))
        j = max(0, int(round((w - tw) / 2.0)))

        cropped = arr[..., i : i + th, j : j + tw]
        if isinstance(img, _Tensor):
            return _from_numpy(np.ascontiguousarray(cropped))
        return cropped

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class RandomCrop:
    """Crop the given image at a random location."""

    def __init__(self, size: Union[int, Tuple[int, int]], padding: int = 0):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.padding = padding

    def __call__(self, img):
        arr = img.to_numpy() if isinstance(img, _Tensor) else np.asarray(img)
        if arr.ndim < 2:
            raise ValueError(f"RandomCrop expects an array with at least 2 dimensions, got shape {arr.shape}")
        if self.padding > 0:
            pad_width = [(0, 0)] * (arr.ndim - 2) + [(self.padding, self.padding), (self.padding, self.padding)]
            arr = np.pad(arr, pad_width, mode="constant")

        th, tw = self.size
        h, w = arr.shape[-2], arr.shape[-1]
        if th > h or tw > w:
            raise ValueError(f"Requested crop size ({th}, {tw}) is larger than input size ({h}, {w})")
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = np.random.randint(0, max(1, h - th + 1))
            j = np.random.randint(0, max(1, w - tw + 1))

        cropped = arr[..., i : i + th, j : j + tw]
        if isinstance(img, _Tensor):
            return _from_numpy(np.ascontiguousarray(cropped))
        return cropped

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"


class Resize:
    """Resizes the given image to the target size using bilinear interpolation."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, img):
        arr = img.to_numpy() if isinstance(img, _Tensor) else np.asarray(img)
        th, tw = self.size
        h, w = arr.shape[-2], arr.shape[-1]

        y_indices = np.linspace(0, h - 1, th).astype(np.float32)
        x_indices = np.linspace(0, w - 1, tw).astype(np.float32)

        y_low = np.floor(y_indices).astype(int)
        y_high = np.clip(y_low + 1, 0, h - 1)
        y_weight = (y_indices - y_low).reshape(-1, 1)

        x_low = np.floor(x_indices).astype(int)
        x_high = np.clip(x_low + 1, 0, w - 1)
        x_weight = (x_indices - x_low).reshape(1, -1)

        # Fully vectorized bilinear interpolation across all channels simultaneously
        channels = arr.reshape(-1, h, w)
        y_grid = y_low[:, None]
        x_grid = x_low[None, :]
        x_grid_h = x_high[None, :]
        y_grid_h = y_high[:, None]

        top = (1.0 - x_weight) * channels[:, y_grid, x_grid] + x_weight * channels[:, y_grid, x_grid_h]
        bottom = (1.0 - x_weight) * channels[:, y_grid_h, x_grid] + x_weight * channels[:, y_grid_h, x_grid_h]
        out_c = (1.0 - y_weight) * top + y_weight * bottom

        out_shape = arr.shape[:-2] + (th, tw)
        out_arr = out_c.reshape(out_shape).astype(np.float32)
        if isinstance(img, _Tensor):
            return _from_numpy(np.ascontiguousarray(out_arr))
        return out_arr

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class Lambda:
    """Applies a user-defined lambda function as a transform."""

    def __init__(self, lambd: Callable):
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ColorJitter:
    """Randomly change the brightness, contrast, and saturation of an image."""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.seed = seed

    def __call__(self, img):
        rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
        arr = img.to_numpy().copy() if isinstance(img, _Tensor) else np.asarray(img, dtype=np.float32).copy()

        # Brightness adjustment
        if self.brightness > 0.0:
            factor = rng.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
            arr = arr * factor

        # Contrast adjustment
        if self.contrast > 0.0:
            factor = rng.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
            mean = float(np.mean(arr))
            arr = (arr - mean) * factor + mean

        arr = np.clip(arr, 0.0, 1.0 if np.max(arr) <= 1.0 else 255.0)
        if isinstance(img, _Tensor):
            return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        return arr

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(brightness={self.brightness}, "
            f"contrast={self.contrast}, saturation={self.saturation}, hue={self.hue})"
        )


__all__ = [
    "Compose",
    "ToTensor",
    "Normalize",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "CenterCrop",
    "RandomCrop",
    "Resize",
    "Lambda",
    "ColorJitter",
]
