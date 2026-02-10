"""SUN397 dataset loader."""

from pathlib import Path
from typing import Optional

from torchvision.datasets import SUN397
from torch.utils.data import Subset

from exrep.utils.data import MultiprocessingManager, IDDataset

class SUN397Dataset(IDDataset, Subset):
    """Wrapper around the SUN397 dataset to load the official split.

    Also supports multiprocessing."""
    def __init__(
        self, split: str, root: str | Path,
        *args, manager: Optional[MultiprocessingManager]=None, **kwargs
    ):
        self.manager = manager

        # initialize the base dataset
        dataset = SUN397(root, *args, **kwargs)

        # get subset indices
        orig_name_to_idx = {
            str(Path(*name.relative_to(root).parts[1:])) : idx
            for idx, name in enumerate(dataset._image_files)
        }

        # TODO: move these to the data directory
        with open(Path(root) / 'SUN397' / f'{split}.txt', 'r', encoding='utf-8') as f:
            self.filenames = f.read().splitlines()

        indices = [orig_name_to_idx[name] for name in self.filenames]

        # initialize the base class
        super().__init__(dataset, indices)

        if self.manager:
            dataset._image_files, dataset._labels, self.indices = self.manager.share(
                dataset._image_files, dataset._labels, self.indices
            )

    def get_input_id(self, idx: int) -> str:
        """Get the image ID (file name) for a given index."""
        return self.filenames[idx]
