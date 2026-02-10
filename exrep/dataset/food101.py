"""Food101 dataset loader."""

from typing import Optional

from torchvision.datasets import Food101

from exrep.utils.data import MultiprocessingManager, IDDataset

class Food101Dataset(Food101, IDDataset):
    """Wrapper around the Food101 dataset to load the official split.

    Also supports multiprocessing."""
    def __init__(
        self, *args, manager: Optional[MultiprocessingManager]=None, **kwargs
    ):
        self.manager = manager

        # initialize the base dataset
        super().__init__(*args, **kwargs)

        if self.manager:
            self._image_files, self._labels = self.manager.share(
                self._image_files, self._labels
            )

    def get_input_id(self: Food101, idx: int) -> str:
        """Get the image ID (file name) for a given index."""
        return str(self._image_files[idx])
