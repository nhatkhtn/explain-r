from pathlib import Path
import random
import logging
import argparse
from multiprocessing import Manager
from typing import Any
from functools import partial

import yaml
from dotenv import dotenv_values, find_dotenv
import wandb
import numpy as np
import torch

from exrep.train import train_bimodal
from exrep.utils.data import prepare_train_val, MultiprocessingManager, get_file_md5
from exrep.dataset.flickr30k import Flickr30kDataset
from exrep.dataset.imagenet import ImageNetDataset
from exrep.dataset.sun397 import SUN397Dataset
from exrep.dataset.coco import CocoDataset
from exrep.dataset.food101 import Food101Dataset
from exrep.dataset.flowers102 import Flowers102Dataset

local_config = dotenv_values(find_dotenv())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASSIFICATION_DATASETS = ['imagenet', 'sun397', 'flowers102', 'food101']

def main(train_config: dict, device, run):
    dataset = train_config['dataset']['name']

    # if evaluating with accuracy, set the dataset to the one used for training
    if train_config['loss']['val']['name'] == 'accuracy':
        train_config['loss']['val']['dataset'] = dataset

    caption_path = None
    train_kwargs: dict[str, Any] = dict(
        split=train_config['dataset'].get('split', 'train')
    )
    val_kwargs: dict[str, Any] = dict(
        split=train_kwargs['split'],
    )
    if dataset == 'coco':
        load_fn = partial(CocoDataset, root=local_config['COCO_ROOT'])  # type: ignore[arg-type]

    elif dataset == 'sun397':
        load_fn = partial(SUN397Dataset, root=local_config['DOWNLOAD_ROOT'])  # type: ignore[arg-type]

    elif dataset == 'flickr30k':
        load_fn = partial(Flickr30kDataset, root=local_config['FLICKR30K_ROOT'])    # type: ignore[arg-type]

    elif dataset == 'imagenet':
        load_fn = partial(ImageNetDataset, root=local_config['IMAGENET_ROOT'])

    elif dataset == 'flowers102':
        load_fn = partial(Flowers102Dataset, root=local_config['DOWNLOAD_ROOT'])  # type: ignore[arg-type]

    elif dataset == 'food101':
        load_fn = partial(Food101Dataset, root=local_config['DOWNLOAD_ROOT']) # type: ignore[arg-type]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if dataset in CLASSIFICATION_DATASETS:
        caption_path = train_config['dataset']['caption_path']

    manager = MultiprocessingManager(Manager())
    train_dataset, val_dataset = prepare_train_val(
        load_fn,
        manager=manager,
        caption_path=caption_path,
        encoding_path=train_config['dataset']['encoding_path'],
        train_augment_sampling=train_config['dataset']['sampling'],
        train_kwargs=train_kwargs,
        val_kwargs=val_kwargs,
    )
    logger.info("Loaded dataset: %s", dataset)

    model, logs = train_bimodal(
        model_config=train_config['surrogate'],
        loss_config=train_config['loss'],
        optimizer_config=train_config['optimizer'],
        target_config=train_config['target'],
        training_config=train_config['training'],
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_path=train_config['output']['path'],
        wandb_run=run,
        log_every_n_steps=1,
        device=device,
    )

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--config', type=str, help='Path to the config file')
    argparse.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    argparse.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    args = argparse.parse_args()

    # set random seed
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    # load the yml config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    run = wandb.init(
        project=local_config["WANDB_PROJECT"],
        config={
            "job_type": "train_representation",
        },
        save_code=True,
    )

    # compute md5
    config['dataset']['encoding_md5'] = get_file_md5(config['dataset']['encoding_path'])
    run.config.update(config)
    main(train_config=config, device=args.device, run=run)
