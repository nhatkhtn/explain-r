"""Takes in the dataset and generates the concept encodings for each image in the dataset."""

import argparse
import logging
import json
from pathlib import Path
import pickle

from tqdm import tqdm
from dotenv import find_dotenv, dotenv_values

from exrep.discover import ImageConceptEncoder, ConceptTensorBuilder, Concept
from exrep.dataset.flickr30k import Flickr30kDataset
from exrep.dataset.sun397 import SUN397Dataset
from exrep.dataset.imagenet import ImageNetDataset
from exrep.dataset.coco import CocoDataset
from exrep.dataset.flowers102 import Flowers102Dataset
from exrep.dataset.food101 import Food101Dataset

local_config = dotenv_values(find_dotenv())
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CLASSIFICATION_DATASETS = ["sun397", "imagenet", "flowers102", "food101"]
SUPPORTED_DATASETS = CLASSIFICATION_DATASETS + ["flickr30k", "coco"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dataset-specific vocabulary and concept encodings for each image."
    )
    parser.add_argument(
        'dataset',
        type=str,
        choices=SUPPORTED_DATASETS,
        help="Name of the dataset to be used.",
    )
    parser.add_argument(
        '-c', '--caption_path',
        type=str,
        help="Path to the caption file. Used by classification datasets. " \
        "If not provided, defaults to 'outputs/{dataset}/captions.json'.",
    )
    parser.add_argument(
        '-vocab', '--vocab_path',
        type=str,
        help="Path to vocabulary. If not provided, defaults to 'outputs/{dataset}/vocab.csv'."
    )
    parser.add_argument(
        '--no_fit_vocab',
        action='store_true',
        help="Whether to fit the vocabulary to the dataset. " \
        "Default is False, i.e. fit the vocabulary when the flag is not present.",
    )
    parser.add_argument(
        '-o', '--output_path',
        type=str,
        help="Path to output pickle file. " \
        "If not provided, defaults to 'outputs/{dataset}/{method}_encoding.pkl'.",
    )
    parser.add_argument(
        "--method",
        type=str, default="average",
        choices=["union", "average"],
        help="Method to use for encoding the concepts. Default is 'average'.",
    )
    parser.add_argument(
        "--min_threshold",
        type=int, default=3,
        help="An integer representing the minimum count for a word to be " \
        "included in the vocabulary. Default is 3.",
    )
    parser.add_argument(
        "--max_threshold",
        type=float, default=0.1,
        help="A float representing the maximum proportion for a word to be " \
        "included in the vocabulary. Default is 0.1.",
    )
    parser.add_argument(
        "--split",
        type=str, default="test",
        help="The split of dataset to explain. Default is 'test'. " \
    )
    args = parser.parse_args()

    # infer default args and check if input files exist
    if args.vocab_path is None:
        args.vocab_path = Path(f"outputs/{args.dataset}/vocab.csv")
    else:
        args.vocab_path = Path(args.vocab_path)

    if args.no_fit_vocab and not args.vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary file {args.vocab_path} does not exist." \
            "Please provide a valid vocabulary file when using the --no_fit_vocab flag."
        )

    if args.output_path is None:
        args.output_path = Path(f"outputs/{args.dataset}/{args.method}_encoding.pkl")
    else:
        args.output_path = Path(args.output_path)

    # create the output directory if it doesn't exist
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset in CLASSIFICATION_DATASETS:
        if args.caption_path is None:
            args.caption_path = Path(f"outputs/{args.dataset}/captions.json")
        else:
            args.caption_path = Path(args.caption_path)

        if not args.caption_path.exists():
            raise FileNotFoundError(
                f"Caption file {args.caption_path} does not exist." \
                "Please provide a valid caption file."
            )

    logger.info("Arguments: %s", args)

    def get_captions(split) -> list[list[str]]:
        """Function to get the captions for a given split."""
        if args.dataset in CLASSIFICATION_DATASETS:
            with open(args.caption_path, 'r', encoding='utf-8') as f:
                captions = json.load(f)

            if args.dataset == "sun397":
                dataset = SUN397Dataset(
                    root=local_config["DOWNLOAD_ROOT"],   # type: ignore[arg-type]
                    split=split,
                )
            elif args.dataset == "imagenet":
                dataset = ImageNetDataset(
                    root=local_config["IMAGENET_ROOT"],   # type: ignore[arg-type]
                    split=split,
                )
            elif args.dataset == "flowers102":
                dataset = Flowers102Dataset(
                    root=local_config["DOWNLOAD_ROOT"],   # type: ignore[arg-type]
                    split=split,
                )
            elif args.dataset == "food101":
                dataset = Food101Dataset(
                    root=local_config["DOWNLOAD_ROOT"],   # type: ignore[arg-type]
                    split=split,
                )
            else:
                raise ValueError(f"Unknown dataset: {args.dataset}. ")
            return [captions[i] for i in dataset.ids()]

        elif args.dataset == "flickr30k":
            dataset = Flickr30kDataset(
                root=local_config["FLICKR30K_ROOT"],   # type: ignore[arg-type]
                split=split,
            )
            return [dataset.annotations[dataset.ids[i]] for i in range(len(dataset))]

        elif args.dataset == "coco":
            dataset = CocoDataset(
                root=local_config["COCO_ROOT"],   # type: ignore[arg-type]
                split=split,
            )
            return [
                dataset._load_target(dataset.ids[i])             # pylint: disable=W0212
                for i in range(len(dataset))
            ]
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}. ")

    train_captions = get_captions(args.split)
    val_split = args.split
    val_captions = get_captions(val_split)

    if args.no_fit_vocab:
        logger.info("Using provided vocabulary without fitting.")
        vocab = Concept.load_vocab_from_csv(args.vocab_path)
        logger.info("Loaded vocab size: %d", len(vocab))
        encoder = ImageConceptEncoder.from_vocabulary(
            vocab, method=args.method,
            min_threshold=args.min_threshold, max_threshold=args.max_threshold
        )
    else:
        logger.info("Fitting dataset vocabulary...")

        encoder = ImageConceptEncoder(
            method=args.method,
            min_threshold=args.min_threshold, max_threshold=args.max_threshold
        )
        encoder.fit(tqdm(train_captions))

        fitted_vocab = encoder.get_vocab()
        logger.info("Fitted vocab size: %d", len(fitted_vocab))
        Concept.save_vocab_to_csv(fitted_vocab, args.vocab_path)
        logger.info("Saved vocab to %s", args.vocab_path)

    logger.info("Encoding the train dataset...")
    train_concepts = encoder.transform(tqdm(train_captions))

    logger.info("Encoding the val dataset...")
    val_concepts = encoder.transform(tqdm(val_captions))


    logger.info("Creating the encodings...")
    tensor_builder = ConceptTensorBuilder(encoder.get_vocab())
    train_encoding = tensor_builder.transform(train_concepts)
    val_encoding = tensor_builder.transform(val_concepts)

    min_train_concepts = min(len(c) for c in train_concepts)

    logger.info("Train encoding shape: %s, nnz: %d, avg sparsity: %.2f, min concept/count: %d/%d",
                train_encoding[2], len(train_encoding[1]),
                len(train_encoding[1]) / train_encoding[2][0],
                min_train_concepts, sum(len(c) == min_train_concepts for c in train_concepts))

    logger.info("Val encoding shape: %s, nnz: %d, avg sparsity: %.2f",
                val_encoding[2], len(val_encoding[1]), len(val_encoding[1]) / val_encoding[2][0])

    logger.info("Saving the encodings...")
    encoding_dict = {
        'train': train_encoding,
        'val': val_encoding,
        'vocab': encoder.get_vocab(),
        'config': vars(args),
    }
    with open(args.output_path, "wb") as output_file:
        pickle.dump(encoding_dict, output_file)
    logger.info("Saved encodings to %s", args.output_path)
