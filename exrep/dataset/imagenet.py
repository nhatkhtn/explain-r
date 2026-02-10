"""ImageNet dataset loading and processing utilities."""

from pathlib import Path
from typing import Optional

from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

from exrep.utils.data import MultiprocessingManager, IDDataset

class ImageNetDataset(IDDataset, Subset):
    """A class to load the ImageNet dataset present in the HPC."""
    def __init__(self,
        root: str | Path, split: str,
        *, manager: Optional[MultiprocessingManager]=None,
        **kwargs
    ):
        root = Path(root)
        self.manager = manager

        # add class names
        with open(root / 'LOC_synset_mapping.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_to_names = dict(line.strip().split(' ', 1) for line in lines)

        dataset = ImageFolder(root / 'Data' / 'CLS-LOC' / split, **kwargs)
        indices = list(range(len(dataset)))

        super().__init__(dataset, indices)

        if self.manager:
            dataset.samples, self.indices = self.manager.share(dataset.samples, self.indices)

    def get_input_id(self, idx: int) -> str:
        """Get the image ID (file name) for a given index."""
        return self.dataset.samples[self.indices[idx]][0].rsplit('/')[-1]
