import argparse
import logging
from functools import partial
from pathlib import Path

from tqdm import tqdm
from dotenv import dotenv_values, find_dotenv
import torch
import clip
from torch.utils.data import DataLoader
import pandas as pd
import splice

from exrep.loss import SimilarityLoss
from exrep.utils.data import prepare_test
from exrep.evaluate import RunningRetrievalStats
from exrep.dataset.flickr30k import Flickr30kDataset
from exrep.dataset.coco import CocoDataset

local_config = dotenv_values(find_dotenv())

class RunningBaselineStats:
    def __init__(self):
        self.total = 0
        self.l0 = 0
        self.cosine = 0

    def update(self, l0s, cosines):
        self.l0 += torch.sum(l0s).item()
        self.cosine += torch.sum(cosines).item()
        self.total += len(l0s)
        return self

    def __str__(self):
        sparsity = self.l0/self.total
        cosine = self.cosine/self.total
        return f"(S/C): ({sparsity:.2f}/{cosine:.2f})"

# run = wandb.init(
#     project=local_config["WANDB_PROJECT"],
#     config={
#         "job_type": "metrics",
#     },
#     save_code=True,
# )
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = ['coco', 'flickr30k']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        type=str,
        choices=SUPPORTED_DATASETS,
        help='Dataset name'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='ViT-B/32',
        help='Target model name',
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        required=True,
        help='Path to the surrogate checkpoint'
    )
    parser.add_argument(
        '--encoding_path',
        type=str,
        help='Path to the encoding file. If not provided, defaults to "outputs/{dataset}/average_encoding.pkl".'
    )
    parser.add_argument(
        '--test_split',
        type=str,
        default='train',
        help="Name of the split to use for evaluation. Default is 'train'."
    )
    parser.add_argument(
        '-l1', '--l1_penalty',
        type=float,
        default=0.25,
        help='L1 penalty for the baseline model'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        default='laion_bigrams',
        help='Vocabulary name for the baseline model'
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=15000,
        help='Vocabulary size for the baseline model'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for evaluation'
    )
    args = parser.parse_args()

    # infer and validate args
    if args.encoding_path is None:
        args.encoding_path = Path(f"outputs/{args.dataset}/average_encoding.pkl")
    else:
        args.encoding_path = Path(args.encoding_path)
    if not args.encoding_path.exists():
        raise ValueError(f"Encoding file {args.encoding_path} does not exist. Please provide a valid encoding file.")

    args.ckpt_path = Path(args.ckpt_path)
    if not args.ckpt_path.exists():
        raise ValueError(f"Checkpoint file {args.ckpt_path} does not exist. Please provide a valid checkpoint file.")

    print(args, flush=True)

    device = args.device
    # load the CLIP model
    target_model, _ = clip.load(args.target, device=device)
    target_model.eval()

    # load our model
    surrogate_model = torch.load(args.ckpt_path, weights_only=False, map_location=device)
    surrogate_model.eval()

    # load splice
    splicemodel = splice.load(f"clip:{args.target}", args.vocab, args.vocab_size, device, l1_penalty=args.l1_penalty, return_weights=True)

    # load the dataset
    if args.dataset == 'flickr30k':
        load_fn = partial(Flickr30kDataset, root=local_config["FLICKR30K_ROOT"])
    elif args.dataset == 'coco':
        load_fn = partial(CocoDataset, root=local_config["COCO_ROOT"])
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}. Supported datasets are: {SUPPORTED_DATASETS}")

    dataset = prepare_test(
        load_fn,
        encoding_path=args.encoding_path,
        transform_captions=True,
        split=args.test_split,
    )

    logger.info("Length of dataset: %d", len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        pin_memory_device=device,
    )

    stats = RunningRetrievalStats()
    baseline_stats = RunningBaselineStats()
    ours_stats = RunningBaselineStats()

    t = 1.0       # temperature doesn't matter for retrieval
    pbar = tqdm(dataloader)
    with torch.inference_mode():
        for batch in pbar:
            images = batch["inputs"].to(device)
            texts = batch["captions"].to(device)
            encodings = batch["encodings"].to(device=device, dtype=torch.float32)

            image_features = target_model.encode_image(images).float()
            text_features = target_model.encode_text(texts).float()

            ours_embeddings = surrogate_model.encode_query(encodings)

            weights = splicemodel.encode_image(images)
            image_embedding_baseline = splicemodel.recompose_image(weights)

            target_sim = SimilarityLoss.get_scaled_sim(image_features, text_features, t)
            student_sim = SimilarityLoss.get_scaled_sim(ours_embeddings, text_features, t)
            baseline_sim = SimilarityLoss.get_scaled_sim(image_embedding_baseline, text_features, t)

            baseline_cosine_matrix = SimilarityLoss.get_scaled_sim(image_embedding_baseline, image_features, t)
            ours_cosine_matrix = SimilarityLoss.get_scaled_sim(ours_embeddings, image_features, t)

            df = stats.update(student_sim, baseline_sim, target_sim)
            baseline_stats.update(torch.linalg.norm(weights, ord=0, dim=1), torch.diag(baseline_cosine_matrix))
            ours_stats.update(torch.linalg.norm(encodings, ord=0, dim=1), torch.diag(ours_cosine_matrix))
            pbar.set_postfix_str(f"Stats: {ours_stats}, Baseline: {baseline_stats}")

    # force pandas to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print("Ours: ", ours_stats)
    print("Baseline: ", baseline_stats)
    print(stats.summary())
