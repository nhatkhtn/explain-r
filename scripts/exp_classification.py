import logging
import argparse
from functools import partial
from multiprocessing import Manager

from dotenv import dotenv_values, find_dotenv
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import clip

import splice
from exrep.utils.data import prepare_test, MultiprocessingManager
from exrep.dataset.sun397 import SUN397Dataset
from exrep.dataset.imagenet import ImageNetDataset
from exrep.dataset.flowers102 import Flowers102Dataset
from exrep.dataset.food101 import Food101Dataset
from exrep.evaluate import get_classes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RunningStats:
    """Convenience class to keep track of running statistics for use in tqdm."""
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.l0 = 0
        self.cosine = 0
        self.faithfulness = 0

    def update(self, preds, target_preds, labels, l0s, cosines):
        self.correct += torch.sum((preds == labels)).item()
        self.faithfulness += torch.sum((preds == target_preds)).item()
        self.l0 += torch.sum(l0s).item()
        self.cosine += torch.sum(cosines).item()
        self.total += preds.shape[0]
        return self

    def __str__(self):
        if self.total != 0:
            acc = self.correct/self.total * 100
            faithfulness = self.faithfulness/self.total * 100
            sparsity = self.l0/self.total
            cosine = self.cosine/self.total
            return f"(A/F/S/C): ({acc:.2f}/{faithfulness:.2f}/{sparsity:.2f}/{cosine:.2f})"
        else:
            return "N/A"

@torch.inference_mode()
def zero_shot_eval(clip_model, splicemodel, model, dataloader, label_embeddings, device):
    """zero_shot_eval Runs zero shot evaluation over a dataloader

    Parameters
    ----------
    splicemodel : splice.SpLiCE
        A SpLiCE model
    model : torch.nn.Module
        Our model
    dataloader : torch.utils.data.Dataloader
        Dataloader to run eval over
    label_embeddings : torch.tensor
        A {num_labels x CLIP dimensionality} tensor of zero-shot label embeddings for each class ("A photo of a {}").

    Returns
    -------
    avg_accuracy, avg_sparsity, avg_cosine_similarity
    """
    our_stats = RunningStats()
    baseline_stats = RunningStats()
    if splicemodel: splicemodel.eval()
    model.eval()
    clip_model.eval()

    clip_correct = 0

    pbar = tqdm(dataloader)
    for batch in pbar:
        image = batch['inputs'].to(device)
        label = batch['label'].to(device)
        encodings = batch['encodings'].to(device, dtype=torch.float32)

        # CLIP inference
        original_embedding = clip_model.encode_image(image).float()
        original_embedding = torch.nn.functional.normalize(original_embedding, dim=1)
        target_preds = find_closest(original_embedding, label_embeddings)
        clip_correct += torch.sum((target_preds == label)).item()

        # baseline inference
        if splicemodel:
            weights = splicemodel.encode_image(image)
            embedding = splicemodel.recompose_image(weights)
            baseline_cosine = torch.nn.functional.cosine_similarity(embedding, original_embedding, dim=1)
            preds = find_closest(embedding, label_embeddings)
            baseline_stats.update(preds, target_preds, label, torch.sum(torch.linalg.norm(weights,ord=0,dim=1)), baseline_cosine)

        # ours inference
        ours_embeddings = model.encode_query(encodings)
        ours_embeddings = torch.nn.functional.normalize(ours_embeddings, dim=1)
        ours_cosine = torch.nn.functional.cosine_similarity(ours_embeddings, original_embedding)
        ours_preds = find_closest(ours_embeddings, label_embeddings)
        our_stats.update(ours_preds, target_preds, label, torch.sum(torch.linalg.norm(encodings,ord=0,dim=1)), ours_cosine)

        pbar.set_postfix_str(f"Ours: {our_stats}, Baseline: {baseline_stats}")

    logger.info("CLIP Accuracy: %s", clip_correct / our_stats.total * 100)

@torch.inference_mode()
def find_closest(embedding, label_embeddings):
    dot_product = embedding@label_embeddings.T
    return torch.argmax(dot_product, dim=-1)

SUPPORTED_DATASETS = ['sun397', 'imagenet', 'flowers102', 'food101']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="sun397", choices=SUPPORTED_DATASETS, help="Dataset to use")
    parser.add_argument('--target', type=str, default="clip:ViT-B/32", help="Target model to use")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the surrogate checkpoint")
    parser.add_argument('--encoding_path', type=str, help="Path to the encoding. If not provided, defaults to outputs/{dataset}/average_encoding.pkl")
    parser.add_argument('--no_baseline', action='store_true', help="Whether to run the baseline. Default is True.")
    parser.add_argument('-l1', '--l1_penalty', type=float, default=0.25, help="L1 penalty for the baseline model")
    parser.add_argument('--vocab', type=str, default="laion_bigrams", help="Vocabulary name for the baseline model")
    parser.add_argument('--vocab_size', type=int, default=15000, help="Vocabulary size for the baseline model")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size for evaluation")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for evaluation")
    args = parser.parse_args()
    
    if args.encoding_path is None:
        args.encoding_path = f"outputs/{args.dataset}/average_encoding.pkl"

    print(args, flush=True)

    # load clip model
    clip_model, _ = clip.load(args.target.split(":")[1], jit=True)

    ## Load SpLiCE Components
    if not args.no_baseline:
        preprocess = splice.get_preprocess(args.target)
        tokenizer = splice.get_tokenizer(args.target)
        splicemodel = splice.load(args.target, args.vocab, args.vocab_size, args.device, l1_penalty=args.l1_penalty, return_weights=True)
        vocab = splice.get_vocabulary(args.vocab, args.vocab_size)
    else:
        splicemodel = None

    ## Load dataset
    local_config = dotenv_values(find_dotenv())
    test_split = 'test'
    if args.dataset == "sun397":
        load_fn = partial(SUN397Dataset, root=local_config["DOWNLOAD_ROOT"])
    elif args.dataset == "imagenet":
        load_fn = partial(ImageNetDataset, root=local_config["IMAGENET_ROOT"])
        test_split = 'val'
    elif args.dataset == "flowers102":
        load_fn = partial(Flowers102Dataset, root=local_config["DOWNLOAD_ROOT"])
    elif args.dataset == "food101":
        load_fn = partial(Food101Dataset, root=local_config["DOWNLOAD_ROOT"])
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    manager = MultiprocessingManager(Manager())
    dataset = prepare_test(load_fn, args.encoding_path, manager=manager, split=test_split)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        pin_memory_device=args.device,
    )

    classes = get_classes(args.dataset)
    prompts = clip.tokenize([f"a photo of a {c}." for c in classes]).to(args.device)

    # obtain CLIP embeddings for the classes
    clip_model.eval()
    with torch.inference_mode():
        label_embeddings = clip_model.encode_text(prompts).to(args.device, dtype=torch.float32)
        label_embeddings = torch.nn.functional.normalize(label_embeddings, dim=1)

    logger.info("Label embeddings shape: %s", label_embeddings.shape)

    # construct our model
    model = torch.load(args.ckpt_path, weights_only=False).to(args.device)
    model.eval()

    zero_shot_eval(clip_model, splicemodel, model, dataloader, label_embeddings, args.device)

if __name__ == "__main__":
    main()
