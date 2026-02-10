"""Perform captioning on a dataset using a pre-trained model."""

import json
import argparse
import logging
from pathlib import Path

from tqdm import tqdm
from dotenv import dotenv_values, find_dotenv
from torch.utils.data import DataLoader

from exrep.caption import CaptioningModel
from exrep.utils.data import TupleToDictOutputDataset, DictOutputDataset

local_config = dotenv_values(find_dotenv())
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROMPT = "a photo of"
SUPPORTED_DATASETS = ["imagenet", "sun397", "flowers102", "food101"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform captioning on a dataset.")
    parser.add_argument(
        "dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        help="The dataset to use for captioning.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="The split of the dataset to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size to use for captioning. Defaults to 32.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="The number of workers used for data loading. Defaults to 1.",
    )
    parser.add_argument(
        "--num-captions-per-image",
        type=int,
        default=10,
        help="The number of captions to generate per image. Defaults to 10.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature to use for sampling. Defaults to 1.0.",
    )
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        help="The path to save the output captions. " \
        "Defaults to outputs/{dataset}/captions.json"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to use for the captioning model. Defaults to cuda.",
    )
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"outputs/{args.dataset}/captions.json"

    logger.info("Arguments: %s", args)

    model = CaptioningModel()
    def transform(x):
        """Wrapper around the HF processor that removes the batch dimension."""
        x = model.processor(images=x, text=PROMPT, return_tensors='pt')
        for k in x.keys():
            x[k] = x[k].squeeze(0)
        return x

    if args.dataset == 'imagenet':
        from exrep.dataset.imagenet import ImageNetDataset
        dataset = ImageNetDataset(
            root=local_config["IMAGENET_ROOT"],   # type: ignore[arg-type]
            split=args.split,
            transform=transform,
        )
    elif args.dataset == 'sun397':
        from exrep.dataset.sun397 import SUN397Dataset
        dataset = SUN397Dataset(
            root=local_config["DOWNLOAD_ROOT"],   # type: ignore[arg-type]
            split=args.split,
            transform=transform,
        )
    elif args.dataset == 'flowers102':
        from exrep.dataset.flowers102 import Flowers102Dataset
        dataset = Flowers102Dataset(
            root=local_config["DOWNLOAD_ROOT"],
            split=args.split,
            transform=transform,
            download=True,
        )
    elif args.dataset == 'food101':
        from exrep.dataset.food101 import Food101Dataset
        dataset = Food101Dataset(
            root=local_config["DOWNLOAD_ROOT"],
            split=args.split,
            transform=transform,
            download=True,
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    dataset = DictOutputDataset(
        TupleToDictOutputDataset(dataset, keys=('inputs', 'label')),
        input_ids=list(dataset.ids()),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        pin_memory_device=args.device,
    )

    captions = {}
    for batch in tqdm(dataloader):
        image_ids = batch['input_ids']
        inputs = batch['inputs'].to(args.device)
        captions.update(dict(
            zip(image_ids,
                model(inputs,
                    num_captions_per_image=args.num_captions_per_image,
                    temperature=args.temperature,
                )
            )
        ))

    # save the captions
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=2)
