"""Flickr30k dataset wrapper."""
from pathlib import Path
from typing import Optional

from torchvision.datasets import Flickr30k

from exrep.utils.data import MultiprocessingManager

class Flickr30kDataset(Flickr30k):
    """Wrapper around the Flickr30k dataset to support multiprocessing."""
    def __init__(
        self, root: str | Path, split: str,
        *args, manager: Optional[MultiprocessingManager]=None, **kwargs
    ):
        root = Path(root)
        self.manager = manager

        super().__init__(
            root, ann_file=root / f'flickr30k_{split}.tsv',     # type: ignore[arg-type]
            *args, **kwargs,
        )

        if self.manager:
            self.annotations, self.ids = self.manager.share(
                self.annotations, self.ids
            )
