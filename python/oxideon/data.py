"""Dataset and DataLoader pipeline for Oxideon.

Provides `Dataset`, `TensorDataset`, `DataLoader`, samplers,
and dataset splitting helpers (`random_split`).
"""

import math
import random
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from . import Tensor, stack


class Dataset:
    """An abstract class representing an indexable or iterable dataset."""

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Any]:
        for idx in range(len(self)):
            yield self[idx]

    def map(self, fn: Callable[[Any], Any]) -> "Dataset":
        """Applies `fn` lazily to each element of this dataset."""
        if isinstance(self, IterableDataset):
            return _MappedIterableDataset(self, fn)
        return MappedDataset(self, fn)

    def filter(self, predicate: Callable[[Any], bool]) -> "FilteredDataset":
        """Lazily retains only elements where `predicate(item)` evaluates to True."""
        return FilteredDataset(self, predicate)

    def cache(self) -> "Dataset":
        """Caches dataset elements in memory after their first access."""
        if isinstance(self, IterableDataset):
            return _CachedIterableDataset(self)
        return CachedDataset(self)

    def repeat(self, count: Optional[int] = None) -> "Dataset":
        """Repeats this dataset `count` times (or indefinitely as an `IterableDataset` if `count is None`)."""
        if isinstance(self, IterableDataset) or count is None:
            return _RepeatedIterableDataset(self, count)
        return RepeatedDataset(self, count)

    def take(self, count: int) -> "Dataset":
        """Limits this dataset to at most the first `count` elements."""
        if isinstance(self, IterableDataset):
            return _TakenIterableDataset(self, count)
        return TakenDataset(self, count)

    def skip(self, count: int) -> "Dataset":
        """Skips the first `count` elements of this dataset."""
        if isinstance(self, IterableDataset):
            return _SkippedIterableDataset(self, count)
        return SkippedDataset(self, count)

    def window(
        self,
        size: int,
        shift: Optional[int] = None,
        stride: int = 1,
        drop_remainder: bool = True,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    ) -> "Dataset":
        """Groups consecutive dataset elements into sliding windows of length `size`."""
        if isinstance(self, IterableDataset):
            return _WindowedIterableDataset(
                self, size=size, shift=shift, stride=stride, drop_remainder=drop_remainder, collate_fn=collate_fn
            )
        return WindowedDataset(
            self, size=size, shift=shift, stride=stride, drop_remainder=drop_remainder, collate_fn=collate_fn
        )


class TensorDataset(Dataset):
    """Dataset wrapping tensors."""

    def __init__(self, *tensors: Tensor):
        assert len(tensors) > 0, "TensorDataset requires at least one tensor."
        first_len = tensors[0].shape[0] if tensors[0].shape else 0
        for i, t in enumerate(tensors):
            cur_len = t.shape[0] if t.shape else 0
            assert cur_len == first_len, (
                f"Size mismatch: tensor {i} has length {cur_len}, expected {first_len}."
            )
        self.tensors = tensors

    def __getitem__(self, index: int) -> Union[Tensor, Tuple[Tensor, ...]]:
        items = tuple(t[index] for t in self.tensors)
        return items[0] if len(items) == 1 else items

    def __len__(self) -> int:
        return self.tensors[0].shape[0] if self.tensors[0].shape else 0


class Subset(Dataset):
    """Subset of a dataset at specified indices."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)


def to_tensors(data: Any) -> Any:
    """Recursively converts NumPy arrays and numeric scalars inside nested structures into Oxideon `Tensor`s
    without adding a batch dimension.
    """
    if isinstance(data, Tensor):
        return data
    if isinstance(data, np.ndarray):
        if data.dtype.kind in ("S", "U", "O"):
            return data
        return Tensor.from_numpy(np.ascontiguousarray(data))
    if isinstance(data, (np.bool_, np.number)):
        return Tensor.from_numpy(np.asarray(data))
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return type(data)(*(to_tensors(d) for d in data))
    if isinstance(data, (tuple, list)):
        return type(data)(to_tensors(d) for d in data)
    if isinstance(data, dict):
        return {k: to_tensors(v) for k, v in data.items()}
    return data


def default_collate(batch: List[Any]) -> Any:
    """Collates a list of samples into a batch tensor or tuple of tensors."""
    if not batch:
        return batch

    elem = batch[0]
    if isinstance(elem, (bool, np.bool_)):
        return Tensor.from_numpy(np.array(batch, dtype=np.bool_))
    if isinstance(elem, Tensor):
        return stack(batch, dim=0)
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return type(elem)(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, (tuple, list)):
        transposed = zip(*batch)
        return type(elem)(default_collate(samples) for samples in transposed)
    elif isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, (int, float)):
        return Tensor.from_numpy(np.array(batch))
    elif isinstance(elem, np.ndarray):
        return Tensor.from_numpy(np.stack(batch, axis=0))
    return batch


class IterableDataset(Dataset):
    """An abstract class representing an Iterable Dataset."""

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("IterableDataset does not support __getitem__")

    def __len__(self) -> int:
        raise TypeError(f"object of type '{type(self).__name__}' has no len()")


class ConcatDataset(Dataset):
    """Dataset as a concatenation of multiple datasets."""

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"
        self.cumulative_sizes = []
        cum = 0
        for d in self.datasets:
            cum += len(d)
            self.cumulative_sizes.append(cum)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Any:
        import bisect
        if idx < 0:
            if -idx > len(self):
                raise IndexError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        if idx >= len(self):
            raise IndexError(
                f"ConcatDataset index {idx} out of range for length {len(self)}"
            )
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class ChainDataset(IterableDataset):
    """Dataset for chaining multiple IterableDataset."""

    def __init__(self, datasets: Sequence[IterableDataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)

    def __iter__(self) -> Iterator[Any]:
        for d in self.datasets:
            for item in d:
                yield item


class Sampler:
    """Base class for all Samplers."""

    def __init__(self, data_source: Optional[Any] = None) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order."""

    def __init__(self, data_source: Any) -> None:
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Samples elements randomly."""

    def __init__(
        self,
        data_source: Any,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.replacement:
            for _ in range(self.num_samples):
                yield self._rng.randint(0, n - 1)
        else:
            indices = list(range(n))
            self._rng.shuffle(indices)
            if self._num_samples is not None:
                indices = indices[: self._num_samples]
            for idx in indices:
                yield idx

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement."""

    def __init__(self, indices: Sequence[int], seed: Optional[int] = None) -> None:
        self.indices = list(indices)
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def __iter__(self) -> Iterator[int]:
        shuffled = list(self.indices)
        self._rng.shuffle(shuffled)
        return iter(shuffled)

    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    """Samples elements from `[0,..,len(weights)-1]` with given probabilities (weights)."""

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer, got {num_samples}")
        w = list(weights)
        if not w:
            raise ValueError("weights should be a non-empty sequence")
        for x in w:
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise TypeError(f"weights must be real numbers, got {type(x).__name__}")
            if math.isnan(x) or math.isinf(x):
                raise ValueError(f"weights must be finite, got {x}")
            if x < 0:
                raise ValueError(f"weights must be non-negative, got {x}")
        if sum(w) <= 0.0:
            raise ValueError("sum of weights must be positive")
        if not replacement and num_samples > len(w):
            raise ValueError(
                f"num_samples ({num_samples}) cannot exceed len(weights) ({len(w)}) without replacement"
            )
        self.weights = w
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed
        self._py_rng = random.Random(seed) if seed is not None else random.Random()
        self._np_rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[int]:
        if self.replacement:
            indices = self._py_rng.choices(range(len(self.weights)), weights=self.weights, k=self.num_samples)
            return iter(indices)
        else:
            weights = np.array(self.weights, dtype=np.float64)
            keys = self._np_rng.exponential(1.0 / np.maximum(weights, 1e-12))
            indices = np.argsort(keys)[: self.num_samples].tolist()
            return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices."""

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer, got {batch_size}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


class DataLoader:
    """Data loader combining a dataset and a sampler, providing an iterable over the given dataset."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Any] = None,
        batch_sampler: Optional[Any] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        if batch_sampler is not None:
            if batch_size != 1 and batch_size is not None or shuffle or sampler is not None or drop_last:
                raise ValueError("batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last")
            batch_size = None
            drop_last = False

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")

        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn is not None else default_collate
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self._is_iterable = isinstance(dataset, IterableDataset)

        if self._is_iterable:
            if shuffle:
                raise ValueError("DataLoader with IterableDataset does not support shuffle=True")
            if sampler is not None or batch_sampler is not None:
                raise ValueError("DataLoader with IterableDataset does not support sampler or batch_sampler")
            self.sampler = None
            self.batch_sampler = None
        else:
            if batch_sampler is None:
                if sampler is None:
                    if shuffle:
                        sampler = RandomSampler(dataset, seed=seed)
                    else:
                        sampler = SequentialSampler(dataset)
                batch_sampler = BatchSampler(sampler, batch_size if batch_size is not None else 1, drop_last)

            self.sampler = sampler
            self.batch_sampler = batch_sampler

    def __len__(self) -> int:
        if self._is_iterable:
            n = len(self.dataset)
            b = self.batch_size if self.batch_size is not None else 1
            return (n // b) if self.drop_last else math.ceil(n / b)
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator[Any]:
        if self._is_iterable:
            b = self.batch_size
            if b is None:
                for item in self.dataset:
                    yield item
                return
            buf: List[Any] = []
            for item in self.dataset:
                buf.append(item)
                if len(buf) == b:
                    yield self.collate_fn(buf)
                    buf = []
            if buf and not self.drop_last:
                yield self.collate_fn(buf)
            return

        if (
            isinstance(self.dataset, TensorDataset)
            and self.collate_fn is default_collate
            and isinstance(self.sampler, (SequentialSampler, RandomSampler))
            and not isinstance(self.dataset, IterableDataset)
        ):
            n = len(self.dataset)
            b = self.batch_size if self.batch_size is not None else 1
            if not self.shuffle and isinstance(self.sampler, SequentialSampler):
                for i in range(0, n, b):
                    if i + b > n and self.drop_last:
                        break
                    end = min(i + b, n)
                    batch_t = tuple(t[i:end] for t in self.dataset.tensors)
                    yield batch_t[0] if len(batch_t) == 1 else batch_t
                return

        for batch_indices in self.batch_sampler:
            batch_samples = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch_samples)


def random_split(
    dataset: Dataset,
    lengths: Sequence[Union[int, float]],
    seed: Optional[int] = None,
) -> List[Subset]:
    """Randomly split a dataset into non-overlapping new datasets of given lengths."""
    total_len = len(dataset)
    if all(isinstance(x, float) for x in lengths) and math.isclose(sum(lengths), 1.0):
        int_lengths = [int(f * total_len) for f in lengths]
        int_lengths[-1] = total_len - sum(int_lengths[:-1])
    else:
        int_lengths = [int(x) for x in lengths]
        if sum(int_lengths) != total_len:
            raise ValueError(
                f"Sum of input lengths ({sum(int_lengths)}) does not equal total dataset length ({total_len})."
            )

    rng = random.Random(seed) if seed is not None else random.Random()
    indices = list(range(total_len))
    rng.shuffle(indices)

    subsets = []
    offset = 0
    for length in int_lengths:
        subsets.append(Subset(dataset, indices[offset : offset + length]))
        offset += length
    return subsets


# ---------------------------------------------------------------------------
# Synthetic Benchmark Dataset Generators
# ---------------------------------------------------------------------------

def make_moons(n_samples: int = 100, noise: float = 0.1, random_state: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Generates two interleaving half circles 2D classification dataset."""
    rng = np.random.RandomState(random_state)
    n0 = n_samples // 2
    n1 = n_samples - n0

    theta_0 = np.linspace(0, np.pi, n0)
    x0 = np.column_stack([np.cos(theta_0), np.sin(theta_0)])
    y0 = np.zeros(n0, dtype=np.float32)

    theta_1 = np.linspace(0, np.pi, n1)
    x1 = np.column_stack([1.0 - np.cos(theta_1), 0.5 - np.sin(theta_1)])
    y1 = np.ones(n1, dtype=np.float32)

    X = np.vstack([x0, x1]).astype(np.float32)
    y = np.concatenate([y0, y1]).astype(np.float32)

    if noise > 0.0:
        X += rng.normal(0, noise, size=X.shape).astype(np.float32)

    from .oxideon import from_numpy
    return from_numpy(np.ascontiguousarray(X)), from_numpy(np.ascontiguousarray(y))


def make_circles(n_samples: int = 100, factor: float = 0.8, noise: float = 0.1, random_state: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Generates concentric circles 2D classification dataset."""
    rng = np.random.RandomState(random_state)
    n0 = n_samples // 2
    n1 = n_samples - n0

    theta_0 = np.linspace(0, 2 * np.pi, n0, endpoint=False)
    x0 = np.column_stack([np.cos(theta_0), np.sin(theta_0)])
    y0 = np.zeros(n0, dtype=np.float32)

    theta_1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    x1 = np.column_stack([factor * np.cos(theta_1), factor * np.sin(theta_1)])
    y1 = np.ones(n1, dtype=np.float32)

    X = np.vstack([x0, x1]).astype(np.float32)
    y = np.concatenate([y0, y1]).astype(np.float32)

    if noise > 0.0:
        X += rng.normal(0, noise, size=X.shape).astype(np.float32)

    from .oxideon import from_numpy
    return from_numpy(np.ascontiguousarray(X)), from_numpy(np.ascontiguousarray(y))


def make_blobs(n_samples: int = 100, n_features: int = 2, centers: int = 3, cluster_std: float = 1.0, random_state: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Generates isotropic Gaussian blobs for clustering and classification."""
    rng = np.random.RandomState(random_state)
    center_coords = rng.uniform(-10.0, 10.0, size=(centers, n_features)).astype(np.float32)

    samples_per_center = [n_samples // centers] * centers
    for i in range(n_samples % centers):
        samples_per_center[i] += 1

    X_list = []
    y_list = []
    for c, n in enumerate(samples_per_center):
        pts = center_coords[c] + rng.normal(0, cluster_std, size=(n, n_features)).astype(np.float32)
        X_list.append(pts)
        y_list.append(np.full(n, c, dtype=np.float32))

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)

    from .oxideon import from_numpy
    return from_numpy(np.ascontiguousarray(X)), from_numpy(np.ascontiguousarray(y))


def make_regression(n_samples: int = 100, n_features: int = 10, noise: float = 0.1, random_state: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Generates a random regression problem."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    true_weights = rng.randn(n_features).astype(np.float32)
    y = X @ true_weights
    if noise > 0.0:
        y += rng.normal(0, noise, size=y.shape).astype(np.float32)

    from .oxideon import from_numpy
    return from_numpy(np.ascontiguousarray(X)), from_numpy(np.ascontiguousarray(y))


__all__ = [
    "Dataset",
    "IterableDataset",
    "TensorDataset",
    "ConcatDataset",
    "ChainDataset",
    "Subset",
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
    "BatchSampler",
    "DataLoader",
    "default_collate",
    "random_split",
    "make_moons",
    "make_circles",
    "make_blobs",
    "make_regression",
]


class DistributedSampler(Sampler):
    """Gives each of `num_replicas` processes its own share of the dataset indices (process `rank` takes every
    num_replicas-th index of the, optionally shuffled, list). Call `set_epoch(epoch)` at the start of every epoch so the
    shuffle differs between epochs but agrees across processes. Without `drop_last` the list is padded by repeating
    indices so every process gets the same number."""

    def __init__(self, dataset: Any, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None or rank is None:
            try:
                from . import distributed as _dist
                num_replicas = _dist.get_world_size() if num_replicas is None else num_replicas
                rank = _dist.get_rank() if rank is None else rank
            except Exception:
                num_replicas = 1 if num_replicas is None else num_replicas
                rank = 0 if rank is None else rank
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset, self.num_replicas, self.rank = dataset, num_replicas, rank
        self.shuffle, self.seed, self.drop_last, self.epoch = shuffle, seed, drop_last, 0
        n = len(dataset)
        if drop_last and n % num_replicas:
            self.num_samples = math.ceil((n - num_replicas) / num_replicas)
        else:
            self.num_samples = math.ceil(n / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        if self.shuffle:
            indices = np.random.default_rng(self.seed + self.epoch).permutation(n).tolist()
        else:
            indices = list(range(n))
        if not self.drop_last:
            pad = self.total_size - len(indices)
            if pad > 0:
                indices += (indices * math.ceil(pad / max(len(indices), 1)))[:pad]
        else:
            indices = indices[: self.total_size]
        return iter(indices[self.rank: self.total_size: self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch, which seeds the shuffle (so every process shuffles the same way)."""
        self.epoch = epoch


class StackDataset(Dataset):
    """Combines datasets of equal length sample by sample: positional datasets give tuples, keyword datasets dicts
    (`StackDataset(images=a, labels=b)[i] == {"images": a[i], "labels": b[i]}`)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and kwargs:
            raise ValueError("StackDataset takes either positional or keyword datasets, not both")
        sets = list(args) if args else list(kwargs.values())
        if not sets:
            raise ValueError("StackDataset needs at least one dataset")
        n = len(sets[0])
        if any(len(s) != n for s in sets):
            raise ValueError("Size mismatch between datasets")
        self._length = n
        self.datasets = tuple(args) if args else dict(kwargs)

    def __getitem__(self, index: int) -> Any:
        if isinstance(self.datasets, dict):
            return {k: d[index] for k, d in self.datasets.items()}
        return tuple(d[index] for d in self.datasets)

    def __getitems__(self, indices: List[int]) -> List[Any]:
        return [self[i] for i in indices]

    def __len__(self) -> int:
        return self._length


__all__ += [
    "DistributedSampler",
    "StackDataset",
    "to_tensors",
    "MappedDataset",
    "FilteredDataset",
    "CachedDataset",
    "RepeatedDataset",
    "TakenDataset",
    "SkippedDataset",
    "WindowedDataset",
    "padded_collate",
    "stratified_split",
    "BalancedClassSampler",
    "LengthBucketSampler",
    "InfiniteDataLoader",
    "cycle_batches",
]


# ---------------------------------------------------------------------------
# Functional Dataset Pipeline Combinators & Advanced Samplers
# ---------------------------------------------------------------------------

def _normalize_index(idx: int, length: int, cls_name: str) -> int:
    if idx < 0:
        idx += length
    if idx < 0 or idx >= length:
        raise IndexError(f"{cls_name} index {idx} out of range for length {length}")
    return idx


class MappedDataset(Dataset):
    """Lazily transforms elements of an indexable `Dataset` with `fn`."""

    def __init__(self, dataset: Dataset, fn: Callable[[Any], Any]) -> None:
        self.dataset = dataset
        self.fn = fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        idx = _normalize_index(index, len(self.dataset), "MappedDataset")
        return self.fn(self.dataset[idx])


class _MappedIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, fn: Callable[[Any], Any]) -> None:
        self.dataset = dataset
        self.fn = fn

    def __iter__(self) -> Iterator[Any]:
        for item in self.dataset:
            yield self.fn(item)


class FilteredDataset(IterableDataset):
    """Lazily yields elements from `dataset` for which `predicate(element)` is truthy."""

    def __init__(self, dataset: Dataset, predicate: Callable[[Any], bool]) -> None:
        self.dataset = dataset
        self.predicate = predicate

    def __iter__(self) -> Iterator[Any]:
        for item in self.dataset:
            if self.predicate(item):
                yield item

    def collect(self) -> List[Any]:
        """Materializes all filtered elements into a Python list."""
        return list(self)


class CachedDataset(Dataset):
    """Memoizes elements of an indexable `Dataset` in memory upon first access."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        idx = _normalize_index(index, len(self.dataset), "CachedDataset")
        if idx not in self._cache:
            self._cache[idx] = self.dataset[idx]
        return self._cache[idx]

    def clear_cache(self) -> None:
        """Clears all cached items."""
        self._cache.clear()


class _CachedIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset) -> None:
        self.dataset = dataset
        self._items: Optional[List[Any]] = None

    def __iter__(self) -> Iterator[Any]:
        if self._items is None:
            buf: List[Any] = []
            for item in self.dataset:
                buf.append(item)
                yield item
            self._items = buf
        else:
            for item in self._items:
                yield item


class RepeatedDataset(Dataset):
    """Repeats an indexable `Dataset` `count` times sequentially."""

    def __init__(self, dataset: Dataset, count: int = 1) -> None:
        if count < 0:
            raise ValueError(f"RepeatedDataset requires count >= 0, got {count}")
        self.dataset = dataset
        self.count = int(count)

    def __len__(self) -> int:
        return len(self.dataset) * self.count

    def __getitem__(self, index: int) -> Any:
        total = len(self)
        idx = _normalize_index(index, total, "RepeatedDataset")
        base_len = len(self.dataset)
        return self.dataset[idx % base_len]


class _RepeatedIterableDataset(IterableDataset):
    def __init__(self, dataset: Dataset, count: Optional[int] = None) -> None:
        if count is not None and count < 0:
            raise ValueError(f"repeat count must be >= 0 or None, got {count}")
        self.dataset = dataset
        self.count = count

    def __iter__(self) -> Iterator[Any]:
        if self.count is None:
            while True:
                yielded = False
                for item in self.dataset:
                    yielded = True
                    yield item
                if not yielded:
                    return
        else:
            for _ in range(self.count):
                for item in self.dataset:
                    yield item


class TakenDataset(Dataset):
    """Exposes at most the first `count` items of an indexable `Dataset`."""

    def __init__(self, dataset: Dataset, count: int) -> None:
        if count < 0:
            raise ValueError(f"take count must be >= 0, got {count}")
        self.dataset = dataset
        self.count = int(count)

    def __len__(self) -> int:
        return min(self.count, len(self.dataset))

    def __getitem__(self, index: int) -> Any:
        idx = _normalize_index(index, len(self), "TakenDataset")
        return self.dataset[idx]


class _TakenIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, count: int) -> None:
        if count < 0:
            raise ValueError(f"take count must be >= 0, got {count}")
        self.dataset = dataset
        self.count = int(count)

    def __iter__(self) -> Iterator[Any]:
        taken = 0
        if self.count == 0:
            return
        for item in self.dataset:
            yield item
            taken += 1
            if taken >= self.count:
                return


class SkippedDataset(Dataset):
    """Skips the first `count` items of an indexable `Dataset`."""

    def __init__(self, dataset: Dataset, count: int) -> None:
        if count < 0:
            raise ValueError(f"skip count must be >= 0, got {count}")
        self.dataset = dataset
        self.count = int(count)

    def __len__(self) -> int:
        return max(0, len(self.dataset) - self.count)

    def __getitem__(self, index: int) -> Any:
        idx = _normalize_index(index, len(self), "SkippedDataset")
        return self.dataset[self.count + idx]


class _SkippedIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, count: int) -> None:
        if count < 0:
            raise ValueError(f"skip count must be >= 0, got {count}")
        self.dataset = dataset
        self.count = int(count)

    def __iter__(self) -> Iterator[Any]:
        skipped = 0
        for item in self.dataset:
            if skipped < self.count:
                skipped += 1
                continue
            yield item


class WindowedDataset(Dataset):
    """Groups elements of an indexable `Dataset` into sliding windows of `size` elements."""

    def __init__(
        self,
        dataset: Dataset,
        size: int,
        shift: Optional[int] = None,
        stride: int = 1,
        drop_remainder: bool = True,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    ) -> None:
        if size <= 0:
            raise ValueError(f"window size must be > 0, got {size}")
        if stride <= 0:
            raise ValueError(f"window stride must be > 0, got {stride}")
        step = size if shift is None else int(shift)
        if step <= 0:
            raise ValueError(f"window shift must be > 0, got {step}")
        self.dataset = dataset
        self.size = int(size)
        self.shift = step
        self.stride = int(stride)
        self.drop_remainder = bool(drop_remainder)
        self.collate_fn = default_collate if collate_fn is None else collate_fn

        n = len(dataset)
        span = (self.size - 1) * self.stride + 1
        starts: List[int] = []
        pos = 0
        while pos < n:
            if pos + span <= n:
                starts.append(pos)
            elif not self.drop_remainder:
                starts.append(pos)
            else:
                break
            pos += self.shift
        self._starts = starts

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, index: int) -> Any:
        idx = _normalize_index(index, len(self._starts), "WindowedDataset")
        start = self._starts[idx]
        n = len(self.dataset)
        items = [
            self.dataset[pos]
            for pos in range(start, min(n, start + self.size * self.stride), self.stride)
        ]
        return self.collate_fn(items)


class _WindowedIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: IterableDataset,
        size: int,
        shift: Optional[int] = None,
        stride: int = 1,
        drop_remainder: bool = True,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    ) -> None:
        if size <= 0 or stride <= 0:
            raise ValueError("window size and stride must be > 0")
        step = size if shift is None else int(shift)
        if step <= 0:
            raise ValueError("window shift must be > 0")
        self.dataset = dataset
        self.size = int(size)
        self.shift = step
        self.stride = int(stride)
        self.drop_remainder = bool(drop_remainder)
        self.collate_fn = default_collate if collate_fn is None else collate_fn

    def __iter__(self) -> Iterator[Any]:
        span = (self.size - 1) * self.stride + 1
        buf: List[Any] = []
        skip_to_next = 0
        for item in self.dataset:
            if skip_to_next > 0:
                skip_to_next -= 1
                continue
            buf.append(item)
            if len(buf) == span:
                win = buf[:: self.stride][: self.size]
                yield self.collate_fn(win)
                if self.shift >= span:
                    skip_to_next = self.shift - span
                    buf.clear()
                else:
                    buf = buf[self.shift :]
        while buf and not self.drop_remainder:
            win = buf[:: self.stride][: self.size]
            if win:
                yield self.collate_fn(win)
            if self.shift >= len(buf):
                break
            buf = buf[self.shift :]


def padded_collate(
    pad_value: float = 0.0,
    padding_side: str = "right",
) -> Callable[[List[Any]], Any]:
    """Returns a `collate_fn` that pads variable-length 1D+ tensors along axis 0 while collating nested structures."""
    if padding_side not in ("right", "left"):
        raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}")

    def _collate(batch: List[Any]) -> Any:
        if not batch:
            return batch
        elem = batch[0]
        if isinstance(elem, Tensor):
            shapes = [tuple(t.shape) for t in batch]
            if all(s == shapes[0] for s in shapes):
                return stack(batch, dim=0)
            if any(len(s) == 0 for s in shapes):
                raise ValueError("padded_collate cannot mix scalar (0-D) and sequence tensors")
            tail = shapes[0][1:]
            if any(s[1:] != tail for s in shapes):
                raise ValueError(f"padded_collate requires matching trailing dimensions, got {shapes}")
            from .rnn_utils import pad_sequence
            return pad_sequence(batch, batch_first=True, padding_value=pad_value, padding_side=padding_side)
        if isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(_collate(list(samples)) for samples in zip(*batch)))
        if isinstance(elem, (tuple, list)):
            return type(elem)(_collate(list(samples)) for samples in zip(*batch))
        if isinstance(elem, dict):
            return {k: _collate([d[k] for d in batch]) for k in elem}
        return default_collate(batch)

    return _collate


def stratified_split(
    dataset: Dataset,
    labels: Union[Tensor, Sequence[Any]],
    lengths: Sequence[Union[int, float]],
    seed: Optional[int] = None,
) -> List[Subset]:
    """Splits `dataset` into non-overlapping `Subset`s preserving per-class proportions across splits."""
    total_len = len(dataset)
    if isinstance(labels, Tensor):
        y_list = labels.to_numpy().reshape(-1).tolist()
    else:
        y_list = np.asarray(labels).reshape(-1).tolist()

    if len(y_list) != total_len:
        raise ValueError(f"labels length ({len(y_list)}) must match dataset length ({total_len})")
    if len(lengths) == 0:
        raise ValueError("lengths must be non-empty")

    if all(isinstance(x, float) for x in lengths) and math.isclose(sum(lengths), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        proportions = [float(x) for x in lengths]
        target_totals = [int(round(p * total_len)) for p in proportions]
        target_totals[-1] = total_len - sum(target_totals[:-1])
    else:
        target_totals = [int(x) for x in lengths]
        if sum(target_totals) != total_len:
            raise ValueError(
                f"Sum of split lengths ({sum(target_totals)}) does not equal dataset length ({total_len})"
            )
        proportions = [t / max(total_len, 1) for t in target_totals]

    rng = random.Random(seed) if seed is not None else random.Random()
    by_class: dict = {}
    for idx, cls in enumerate(y_list):
        by_class.setdefault(cls, []).append(idx)
    for cls_indices in by_class.values():
        rng.shuffle(cls_indices)

    k = len(target_totals)
    split_buckets: List[List[int]] = [[] for _ in range(k)]

    # Allocate integer floors per class first, then distribute remainders by largest fractional part
    # while respecting exact `target_totals` capacities.
    remainders: List[Tuple[int, List[Tuple[float, int]]]] = []
    for cls, cls_indices in by_class.items():
        m = len(cls_indices)
        exact = [p * m for p in proportions]
        floors = [int(math.floor(v)) for v in exact]
        offset = 0
        for s_idx, cnt in enumerate(floors):
            split_buckets[s_idx].extend(cls_indices[offset : offset + cnt])
            offset += cnt
        leftover = cls_indices[offset:]
        frac_order = sorted(((exact[i] - floors[i], i) for i in range(k)), reverse=True)
        for rem_idx in leftover:
            remainders.append((rem_idx, frac_order))

    rng.shuffle(remainders)
    for sample_idx, pref_splits in remainders:
        placed = False
        for _, s_idx in pref_splits:
            if len(split_buckets[s_idx]) < target_totals[s_idx]:
                split_buckets[s_idx].append(sample_idx)
                placed = True
                break
        if not placed:
            for s_idx in range(k):
                if len(split_buckets[s_idx]) < target_totals[s_idx]:
                    split_buckets[s_idx].append(sample_idx)
                    break

    for bucket in split_buckets:
        rng.shuffle(bucket)

    return [Subset(dataset, bucket) for bucket in split_buckets]


class BalancedClassSampler(Sampler):
    """Samples an equal number of indices from every class per epoch (oversampling minority classes by default)."""

    def __init__(
        self,
        labels: Union[Tensor, Sequence[Any]],
        samples_per_class: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(None)
        if isinstance(labels, Tensor):
            y_list = labels.to_numpy().reshape(-1).tolist()
        else:
            y_list = np.asarray(labels).reshape(-1).tolist()
        if not y_list:
            raise ValueError("BalancedClassSampler requires at least one label")

        groups: dict = {}
        for idx, cls in enumerate(y_list):
            groups.setdefault(cls, []).append(idx)
        self._class_groups: List[List[int]] = [groups[c] for c in sorted(groups.keys(), key=str)]
        max_cnt = max(len(g) for g in self._class_groups)
        spc = max_cnt if samples_per_class is None else int(samples_per_class)
        if spc <= 0:
            raise ValueError(f"samples_per_class must be > 0, got {spc}")
        self.samples_per_class = spc
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def __len__(self) -> int:
        return self.samples_per_class * len(self._class_groups)

    def __iter__(self) -> Iterator[int]:
        chosen: List[int] = []
        for grp in self._class_groups:
            if len(grp) >= self.samples_per_class:
                picked = self._rng.sample(grp, self.samples_per_class)
            else:
                picked = list(grp) + [self._rng.choice(grp) for _ in range(self.samples_per_class - len(grp))]
            chosen.extend(picked)
        self._rng.shuffle(chosen)
        return iter(chosen)


class LengthBucketSampler(Sampler):
    """Batch sampler grouping sequences of similar length into mini-batches to minimize padding overhead."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        drop_last: bool = False,
        bucket_pool_multiplier: int = 25,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(None)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if bucket_pool_multiplier <= 0:
            raise ValueError(f"bucket_pool_multiplier must be > 0, got {bucket_pool_multiplier}")
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.bucket_pool_multiplier = int(bucket_pool_multiplier)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def __len__(self) -> int:
        n = len(self.lengths)
        return (n // self.batch_size) if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        n = len(self.lengths)
        indices = list(range(n))
        if self.shuffle:
            self._rng.shuffle(indices)

        pool_size = self.batch_size * self.bucket_pool_multiplier
        batches: List[List[int]] = []
        for p_start in range(0, n, pool_size):
            pool = indices[p_start : p_start + pool_size]
            pool.sort(key=lambda i: self.lengths[i])
            for b_start in range(0, len(pool), self.batch_size):
                batch = pool[b_start : b_start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle:
            self._rng.shuffle(batches)
        return iter(batches)


class InfiniteDataLoader:
    """Wraps a `DataLoader` (or any non-empty iterable) as a repeatable endless stream of batches."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader
        self.epoch = 0

    def __iter__(self) -> Iterator[Any]:
        while True:
            sampler = getattr(self.loader, "sampler", None)
            if hasattr(sampler, "set_epoch") and callable(sampler.set_epoch):
                sampler.set_epoch(self.epoch)
            yielded = False
            for batch in self.loader:
                yielded = True
                yield batch
            if not yielded:
                return
            self.epoch += 1


def cycle_batches(loader: Any) -> Iterator[Any]:
    """Yields batches endlessly from `loader`, automatically restarting epochs when exhausted."""
    return iter(InfiniteDataLoader(loader))
