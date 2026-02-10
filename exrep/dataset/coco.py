"""Utility functions for loading the COCO dataset."""
from pathlib import Path
from typing import Optional

from torchvision.datasets import CocoCaptions

from exrep.utils.data import MultiprocessingManager

class CocoDataset(CocoCaptions):
    """Wrapper around the COCO 2017 dataset to support multiprocessing."""
    def __init__(
        self, root: str | Path, split: str,
        *args, manager: Optional[MultiprocessingManager]=None, **kwargs
    ):
        root = Path(root)
        self.manager = manager

        super().__init__(
            root=root / '2017' / f'{split}2017',
            annFile=root / '2017' / 'annotations' / f'captions_{split}2017.json',   # type: ignore[arg-type]
            *args, **kwargs,
        )

        if self.manager:
            self.ids, self.coco.imgs, self.coco.anns, self.coco.imgToAnns = self.manager.share(
                self.ids, self.coco.imgs, self.coco.anns, self.coco.imgToAnns
            )
