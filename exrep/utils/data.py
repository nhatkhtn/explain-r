"""Utility functions for data loading and processing."""

import json
import random
import logging
from pathlib import Path
from abc import abstractmethod, ABC
from typing import Optional, Sequence, Callable, Any, Iterator
from functools import reduce
import operator
from fractions import Fraction
import pickle
import hashlib

import torch
from scipy.stats._binomtest import _binom_wilson_conf_int
from torch.utils.data import Dataset, StackDataset

from exrep.utils.preprocess import get_coco_clip_transforms

logger = logging.getLogger(__name__)

classification_keys = ('inputs', 'label')
retrieval_keys = ('inputs', 'captions')

class MultiprocessingManager:
    """A class to create object proxies for multiprocessing."""
    def __init__(self, manager=None):
        self.manager = manager

    def share(self, *args):
        """Convert the given arguments to shared objects."""
        if self.manager is None:
            return args

        shared_args = []
        for arg in args:
            if isinstance(arg, list):
                shared_args.append(self.manager.list(arg))
            elif isinstance(arg, dict):
                shared_args.append(self.manager.dict(arg))
            else:
                logger.warning("Unsupported type %s for multiprocessing manager, ignoring.", type(arg))
                shared_args.append(arg)
        return tuple(shared_args)

DatasetProtocol = Dataset | Sequence | torch.Tensor

class DictOutputDatasetInterface(Dataset):
    """Interface for datasets that return a dictionary."""
    def __len__(self) -> int:                             # type: ignore[return-value]
        """Get the length of the dataset."""
        raise NotImplementedError("Subclasses must implement __len__.")

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[return-value]
        """Get the item at the given index as a dictionary."""
        raise NotImplementedError("Subclasses must implement __getitem__.")

class DictOutputDataset(DictOutputDatasetInterface):
    """A dataset that returns a dictionary of outputs."""
    def __init__(self, *args: DictOutputDatasetInterface, **kwargs: DatasetProtocol):
        if len(args) == 0 and len(kwargs) == 0:
            raise ValueError("At least one dataset must be provided.")

        args = args + (StackDataset(**kwargs), )            # type: ignore[arg-type]
        if any(len(arg) != len(args[0]) for arg in args):
            raise ValueError("All datasets must have the same length.")
        self.datasets = args

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index) -> dict[str, Any]:
        return reduce(operator.ior, [dataset[index] for dataset in self.datasets], {})

    def __getitems__(self, indices: list) -> list[dict[str, Any]]:
        data: list[list[dict]] = []
        for dataset in self.datasets:
            if callable(getattr(dataset, "__getitems__", None)):
                data.append(dataset.__getitems__(indices))  # type: ignore[attr-defined]
            else:
                data.append([dataset[index] for index in indices])
        return [reduce(operator.ior, item, {}) for item in zip(*data)]

class TupleToDictOutputDataset(DictOutputDatasetInterface):
    """Converts a tuple output dataset to a dictionary output dataset."""
    def __init__(self, dataset: DatasetProtocol, keys: tuple[str, ...]):
        self.dataset = dataset
        self.keys = keys
    def __len__(self):
        return len(self.dataset)    # type: ignore[arg-type]
    def __getitem__(self, index):
        datum = self.dataset[index]
        if not isinstance(datum, tuple):
            raise ValueError("The dataset must return a tuple.")
        if len(datum) != len(self.keys):
            raise ValueError("The number of keys must match the number of outputs.")
        return {key: value for key, value in zip(self.keys, datum)}

class SparseEncoding(Dataset):
    """Sparse concept encoding that returns samples from the estimator distribution."""
    def __init__(self, indices, values, size, total_mass=0.8, dtype=torch.float16):
        self.size = size
        self.dtype = dtype
        self.total_mass = total_mass

        self.rows = [[] for _ in range(size[0])]
        for (row, col), value in zip(zip(indices[0], indices[1]), values):
            self.rows[row].append((col, value))

    def __len__(self):
        return self.size[0]

    @property
    def shape(self):
        """Returns the shape as a tuple, mimicking the behavior of a tensor."""
        return self.size

    def _sample_p(self, value: Fraction):
        """Sample p from the estimator distribution."""
        r = random.random() * self.total_mass
        low, high = _binom_wilson_conf_int(
            value.numerator, value.denominator, r, 'two-sided', False
        )
        return random.choice((low, high))

    def __getitem__(self, idx):
        row_vector = torch.zeros(self.size[1], dtype=self.dtype)
        for col, value in self.rows[idx]:
            row_vector[col] = self._sample_p(value)
        return row_vector

    def __getitems__(self, indices):
        tensor = torch.zeros(len(indices), self.size[1], dtype=self.dtype)
        for i, idx in enumerate(indices):
            for col, value in self.rows[idx]:
                tensor[i][col] = self._sample_p(value)
        return tensor

class IDDataset(Dataset, ABC):
    """A dataset that provides an ID for each sample."""

    @abstractmethod
    def get_input_id(self, idx: int) -> str:
        """Get the image ID (file name) for a given index."""
        raise NotImplementedError("Subclasses must implement get_input_id.")

    def ids(self) -> Iterator[str]:
        """Get the image IDs (file names) for all samples, in the order of the dataset."""
        for index in range(len(self)):
            yield self.get_input_id(index)

class LazyTransform(Dataset):
    """A dataset wrapper that applies a transform to the dataset lazily."""
    def __init__(self, dataset: DatasetProtocol, transform: Optional[Callable]=None):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)    # type: ignore[arg-type]
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.transform:
            item = self.transform(item)
        return item

def prepare_encoding(indices, values, size, training: bool):
    """Construct a normal tensor if not training, otherwise a sampled encoding."""
    if not isinstance(values[0], Fraction):
        raise ValueError("Values must be of type Fraction.")

    if not training:
        return torch.sparse_coo_tensor(
            indices, [float(v) for v in values], size,
            dtype=torch.float16,
        ).to_dense()

    return SparseEncoding(
        indices=torch.tensor(indices, dtype=torch.int64),
        values=values,
        size=size,
    )

def augment_captions(
    dataset: DictOutputDatasetInterface,
    captions: Sequence[Sequence[str]],
    caption_transform: Optional[Callable] = None,
):
    """Augment the dataset with captions."""
    if hasattr(dataset, 'manager') and isinstance(dataset.manager, MultiprocessingManager):     # type: ignore[attr-defined]
        captions,  = dataset.manager.share(captions)    # type: ignore[attr-defined]
    return DictOutputDataset(
        dataset,
        captions=LazyTransform(
            captions,
            transform=caption_transform
        )
    )

def prepare_train_val(
    loader_fn: Callable[..., Dataset],
    encoding_path: str | Path,
    manager,
    caption_path: Optional[str | Path] = None,
    train_augment_sampling = False,
    train_kwargs: dict = {},
    val_kwargs: dict = {},
):
    """Utility function to prepare the dataset for training and validation"""
    with open(encoding_path, 'rb') as f:
        encoding_dict = pickle.load(f)
    train_encoding = encoding_dict['train']
    val_encoding = encoding_dict['val']

    logger.info("Encoding config %s", encoding_dict['config'])

    train_encoding = prepare_encoding(*train_encoding, training=train_augment_sampling)
    val_encoding = prepare_encoding(*val_encoding, training=False)

    logger.info("Train encoding shape: %s", train_encoding.shape)
    logger.info("Val encoding shape: %s", val_encoding.shape)

    # load transforms
    # warning: we assume the model input size is 224
    train_transform, train_target_transform, val_transform, val_target_transform = get_coco_clip_transforms(224)

    # optionally augment with captions
    if caption_path is not None:
        with open(caption_path, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        logger.info("Loaded %d captions", len(captions))

        train_dataset = loader_fn(
            manager=manager, transform=train_transform, **train_kwargs,
        )
        val_dataset = loader_fn(
            manager=manager, transform=val_transform, **val_kwargs,
        )

        if not isinstance(train_dataset, IDDataset) or not isinstance(val_dataset, IDDataset):
            raise ValueError("The dataset must be an IDDataset to augment with captions.")

        train_dataset = augment_captions(
            TupleToDictOutputDataset(train_dataset, keys=classification_keys),
            captions=[captions[k] for k in train_dataset.ids()],
            caption_transform=train_target_transform,
        )
        val_dataset = augment_captions(
            TupleToDictOutputDataset(val_dataset, keys=classification_keys),
            captions=[captions[k] for k in val_dataset.ids()],
            caption_transform=val_target_transform,
        )
    else:
        train_dataset = loader_fn(
            manager=manager, transform=train_transform,
            target_transform=train_target_transform,
            **train_kwargs,
        )
        val_dataset = loader_fn(
            manager=manager, transform=val_transform,
            target_transform=val_target_transform,
            **val_kwargs,
        )

        train_dataset = TupleToDictOutputDataset(train_dataset, keys=retrieval_keys)
        val_dataset = TupleToDictOutputDataset(val_dataset, keys=retrieval_keys)

    logger.info("Train dataset size: %d", len(train_dataset))
    logger.info("Val dataset size: %d", len(val_dataset))

    train_merged = DictOutputDataset(
        train_dataset,
        encodings=train_encoding,
        indices=torch.arange(len(train_encoding)),
    )
    val_merged = DictOutputDataset(
        val_dataset,
        encodings=val_encoding,
        indices=torch.arange(len(val_encoding)),
    )
    return train_merged, val_merged

def prepare_test(
    loader_fn: Callable[..., Dataset],
    encoding_path: str | Path,
    transform_captions: bool = False,
    split: str = 'train',
    **kwargs,
):
    """Utility function to prepare the dataset for testing"""
    with open(encoding_path, 'rb') as f:
        encoding_dict = pickle.load(f)
    test_encoding = encoding_dict['train']

    test_encoding = prepare_encoding(*test_encoding, training=False)
    logger.info("Test encoding shape: %s", test_encoding.shape)

    # load dataset
    # warning: we assume the model input size is 224
    _, _, transform, caption_transform = get_coco_clip_transforms(224)
    target_transform = caption_transform if transform_captions else None

    test_dataset = loader_fn(
        split=split,
        transform=transform,
        target_transform=target_transform,
        **kwargs
    )

    logger.info("Test dataset size: %d", len(test_dataset))     # type: ignore[arg-type]

    return DictOutputDataset(
        TupleToDictOutputDataset(
            test_dataset,
            keys=retrieval_keys if transform_captions else classification_keys
        ),
        encodings=test_encoding,
    )

def get_file_md5(filename: str | Path):
    """Calculate the MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(filename, "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def check_encoding_md5(run_name: str, enc_path: str | Path):
    return
    import wandb
    api = wandb.Api()
    runs = api.runs("<>/explain-representation")
    for run in runs:
        if run.name == run_name:
            break
    else:
        raise ValueError("Run not found")

    if get_file_md5(enc_path) != run.config['dataset']['encoding_md5']:
        logger.warning("MD5 hash of encoding file does not match the one in the run config. ")
        logger.warning("The current encoding file is %s, and the one in the run config is %s", enc_path, run.config['dataset']['encoding_md5'])
    else:
        logger.info("Successfully verified checksum of encoding file.")