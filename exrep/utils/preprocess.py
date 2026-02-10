import random
import logging

import torch
import torchvision.transforms.v2 as v2

logger = logging.getLogger(__name__)

# see https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L85C19-L85C92
normalize_transform = v2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

def get_clip_image_transforms(input_size):
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RGB(),
        v2.ToDtype(torch.float32, scale=True),
        normalize_transform,
    ])
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(input_size),
        v2.CenterCrop(input_size),
        v2.RGB(),
        v2.ToDtype(torch.float32, scale=True),
        normalize_transform,
    ])
    return train_transform, val_transform

def get_caption_transforms():
    import clip
    def train_transform(texts):
        caption_choice = random.choice(texts)
        # the clip tokenizer adds a batch dimension
        # so we need to remove it
        return clip.tokenize(caption_choice, truncate=True)[0]
    def val_transform(texts):
        # the val transform takes the first caption and also removes the batch dimension
        return clip.tokenize(texts[0], truncate=True)[0]
    return train_transform, val_transform

def get_coco_clip_transforms(input_size=224):
    train_image_transform, val_image_transform = get_clip_image_transforms(input_size=input_size)
    train_text_transform, val_text_transform = get_caption_transforms()
    return train_image_transform, train_text_transform, val_image_transform, val_text_transform