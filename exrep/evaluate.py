from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Sequence

from dotenv import dotenv_values, find_dotenv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import SUN397, Food101
import clip
import pandas as pd

from exrep.loss import SimilarityLoss
from exrep.model.distill import DistilledRepresentationModel

prompt_template = "a photo of a {}."

def dict_avg(dicts: list[dict]) -> dict[str, float]:
    """Average the values in a list of dictionaries."""
    agg_dict = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            agg_dict[k].append(v)
    return {k: np.mean(v).item() for k, v in agg_dict.items()}

@torch.inference_mode()
def compute_classify_acc(model_sim: torch.Tensor, target_sim: torch.Tensor):
    """Calculate classification accuracy from similarity scores."""
    preds = model_sim.argmax(dim=1)
    targets = target_sim.argmax(dim=1)
    acc = (preds == targets).float().mean().item()
    return acc

def classify_loop(model: DistilledRepresentationModel, val_dataloader: DataLoader, prompts: torch.Tensor, device):
    model.eval()
    prompts = prompts.to(device)

    metrics = []

    temperature = model.surrogate_model.temperature
    for batch in val_dataloader:
        # move to device
        images = batch["inputs"].to(device)
        # texts = texts.to(device)
        encodings = batch["encodings"].to(device=device, dtype=torch.float32)

        # forward pass
        with torch.inference_mode():
            queries_student, keys_student, queries_teacher, keys_teacher = model(images, prompts, encodings)
            student_sim = SimilarityLoss.get_scaled_sim(queries_student, keys_student, temperature)
            teacher_sim = SimilarityLoss.get_scaled_sim(queries_teacher, keys_teacher, temperature)

        metrics_dict = {
            "val_accuracy": compute_classify_acc(student_sim, teacher_sim),
        }
        metrics.append(metrics_dict)

    metrics = dict_avg(metrics)
    return metrics

def get_classes(dataset_name: str):
    """Get the classes (assumed in same order as the labels) for the given dataset."""
    local_config = dotenv_values(find_dotenv())
    if dataset_name == 'sun397':
        clases_names = SUN397(local_config["DOWNLOAD_ROOT"]).classes
    elif dataset_name == 'food101':
        clases_names = Food101(local_config["DOWNLOAD_ROOT"]).classes
    elif dataset_name == 'flowers102':
        # https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt
        class_path = Path(local_config["REPO_ROOT"]) / "data" / "oxford102_labels.txt"
        with open(class_path, "r", encoding='utf-8') as f:
            clases_names = [line.strip("\' \n") for line in f.readlines()]
    elif dataset_name == 'imagenet':
        # from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
        class_path = Path(local_config["REPO_ROOT"]) / "data" / "imagenet_classes.txt"
        with open(class_path, "r", encoding='utf-8') as f:
            clases_names = [line.strip() for line in f.readlines()]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name} for classification evaluator")
    return clases_names

def get_classification_evaluator(dataset_name: str):
    """Get the classification evaluator for the given dataset."""
    classes = get_classes(dataset_name)
    prompts = [prompt_template.format(name) for name in classes]
    return partial(classify_loop, prompts=clip.tokenize(prompts))

# adapted from https://github.com/openai/CLIP/issues/115
def compute_retrieval(a2b_sims, targets):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    assert len(targets) == npts, "Number of targets must be the same as the number of embeddings."
    # loop source embedding indices
    for index, target in enumerate(targets):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == target)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    report_dict = {
        "r1": r1, "r5": r5, "r10": r10, "r50": r50,
        "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10,
    }
    return report_dict

def compute_retrieval_symmetric(
    similarity: np.ndarray, labels_i2t: np.ndarray, labels_t2i: np.ndarray
) -> dict[tuple[str, str], float]:
    return {('i2t', metric): value for metric, value in compute_retrieval(similarity, labels_i2t).items()} | \
           {('t2i', metric): value for metric, value in compute_retrieval(similarity.T, labels_t2i).items()}

@torch.inference_mode()
def explain_retrieval_metrics(student_sim: torch.Tensor, target_sim: torch.Tensor, index: Sequence[str]):
    """Compute explanation metrics given original and surrogate similarity matrices."""
    target_i2t_preds = target_sim.argmax(dim=1).detach().cpu().numpy()
    target_t2i_preds = target_sim.argmax(dim=0).detach().cpu().numpy()
    labels_i2t = np.arange(target_sim.shape[0])
    labels_t2i = np.arange(target_sim.shape[1])

    student_sim_np = student_sim.detach().cpu().numpy()

    student_performance = compute_retrieval_symmetric(student_sim_np, labels_i2t, labels_t2i)
    student_faithfulness = compute_retrieval_symmetric(student_sim_np, target_i2t_preds, target_t2i_preds)

    columns = pd.MultiIndex.from_tuples(student_performance.keys())
    df = pd.DataFrame([
        student_performance.values(),
        student_faithfulness.values(),
    ], index=index, columns=columns)
    return df

class RunningRetrievalStats:
    """Class to compute retrieval metrics."""
    def __init__(self):
        self.stats = []

    def update(self, student_sim: torch.Tensor, baseline_sim: torch.Tensor, target_sim: torch.Tensor):
        assert student_sim.shape == target_sim.shape, "student and target similarity matrices must have the same shape"
        assert student_sim.ndim == 2, "student and target similarity matrices must be 2D"
        assert student_sim.shape[0] == student_sim.shape[1], "student and target similarity matrices must be square"

        assert baseline_sim.shape == target_sim.shape, "baseline and target similarity matrices must have the same shape"
        assert baseline_sim.ndim == 2, "baseline and target similarity matrices must be 2D"
        assert baseline_sim.shape[0] == baseline_sim.shape[1], "baseline and target similarity matrices must be square"

        student_metrics = explain_retrieval_metrics(student_sim, target_sim, ["Ours performance", "Ours faithfulness"])
        baseline_metrics = explain_retrieval_metrics(baseline_sim, target_sim, ["Baseline performance", "Baseline faithfulness"])

        target_metrics = explain_retrieval_metrics(target_sim, target_sim, ["Target performance", "Target faithfulness"])
        # disregard the target faithfulness metric
        target_metrics = target_metrics.iloc[:-1]

        df = pd.concat([student_metrics, baseline_metrics, target_metrics], axis=0)
        self.stats.append(df)
        return df

    def summary(self):
        """Return the summary of the stats."""
        return sum(self.stats) / len(self.stats)
