from pathlib import Path
from typing import Sequence, Optional

from dotenv import dotenv_values, find_dotenv
import torchvision
from torch.utils.data import Dataset, ConcatDataset

from exrep.utils.data import ImageLabelCaptionsDataset

local_config = dotenv_values(find_dotenv())
DOWNLOAD_ROOT = Path(local_config["DOWNLOAD_ROOT"])

class VOCClassificationMultiprocessing(torchvision.datasets.VOCDetection):
    """Wrapper for VOCDetection to support multiprocessing."""
    def __init__(self, *args, **kwargs):
        manager = kwargs.pop('manager', None)
        super().__init__(*args, **kwargs)
        if manager:
            # wrap the dataset for multiprocessing
            self.images = manager.list(self.images)
            self.targets = manager.list(self.targets)
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow', 
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.class_to_idx = {classname: idx for idx, classname in enumerate(self.classes)}

    def __getitem__(self, idx: int):
        image, target = super().__getitem__(idx)
        labels = list(set(self.class_to_idx[anno['name']] for anno in target['annotation']['object']))
        label_onehot = [0] * len(self.classes)
        for label in labels:
            label_onehot[label] = 1
        return image, label_onehot

def load_voc(
    split: str, manager = None, captions: Optional[dict[str, Sequence[Sequence[str]]]] = None,
    return_labels=True, caption_transform=None, **kwargs
) -> Dataset:
    """Load the VOC dataset, with optional captions."""
    if split == 'train':
        dataset = ConcatDataset([
            VOCClassificationMultiprocessing(root=voc_root, year='2007', image_set='trainval', manager=manager, **kwargs),
            VOCClassificationMultiprocessing(root=voc_root, year='2007', image_set='test', manager=manager, **kwargs),
        ])
    else:
        dataset = VOCClassificationMultiprocessing(root=voc_root, year='2007', image_set=split, manager=manager, **kwargs)

    if captions:
        caption_list = captions[split]
        if manager:
            caption_list = manager.list(caption_list)
        dataset = ImageLabelCaptionsDataset(dataset, caption_list, return_labels=return_labels, caption_transform=caption_transform)

    return dataset
