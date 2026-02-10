from pathlib import Path
from functools import partial
import pickle
from typing import Optional

import torch
from torch.utils.data import DataLoader
from splice import SPLICE
from dotenv import dotenv_values, find_dotenv
import clip

from exrep.dataset.imagenet import ImageNetDataset
from exrep.dataset.coco import CocoDataset
from exrep.dataset.sun397 import SUN397Dataset
from exrep.dataset.flowers102 import Flowers102Dataset
from exrep.dataset.food101 import Food101Dataset
from exrep.utils.data import prepare_test, check_encoding_md5
from exrep.model.surrogate import SurrogateRepresentation
from exrep.attribution import attribute_embeddings, get_top_abs

local_config = dotenv_values(find_dotenv())

@torch.inference_mode()
def get_rep_concepts(model: SurrogateRepresentation, vocab, splicemodel: SPLICE, splice_vocab, dataset, k=10, device='cuda'):
    ours_attributions = attribute_embeddings(model, dataset, device)
    
    images = torch.stack([datum['inputs'] for datum in dataset], dim=0).to(device)
    splice_weights = splicemodel.encode_image(images)

    ours_concepts, splice_concepts = [], []
    values, indices = get_top_abs(ours_attributions, dim=1, k=k)

    for i in range(len(dataset)):
        ours_concepts.append([(values[i][j].detach().item(), vocab[indices[i][j]]) for j in range(k)])

        splice_concepts.append([
            (splice_weights[i][j].detach().item(), splice_vocab[j]) for j in torch.sort(splice_weights[i], descending=True)[1][:k]
        ])
    return ours_concepts, splice_concepts

@torch.inference_mode()
def get_predictions(
    model: SurrogateRepresentation,
    splicemodel: SPLICE,
    clip_model,
    dataset,
    text_embeds: torch.Tensor,
    mask: Optional[dict[str, float]] = None,
    device='cuda'
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 
    
    preds_dict = {"ours": [], "clip": [], "splice": []}
    for datum in dataloader:
        datum['inputs'] = datum['inputs'].to(device)
        datum['encodings'] = datum['encodings'].to(device, dtype=torch.float32)

        if mask is not None:
            for k, v in mask.items():
                datum['encodings'][:, k] = v

        ours_embedding = model.encode_query(datum['encodings'])
        ours_embedding = torch.nn.functional.normalize(ours_embedding, dim=-1)
        clip_embedding = clip_model.encode_image(datum['inputs']).float()
        clip_embedding = torch.nn.functional.normalize(clip_embedding, dim=-1)
        splice_weights = splicemodel.encode_image(datum['inputs'])
        splice_embedding = splicemodel.recompose_image(splice_weights)
        splice_embedding = torch.nn.functional.normalize(splice_embedding, dim=-1)

        ours_pred = torch.argmax(ours_embedding @ text_embeds.T, dim=-1)
        clip_pred = torch.argmax(clip_embedding @ text_embeds.T, dim=-1)
        splice_pred = torch.argmax(splice_embedding @ text_embeds.T, dim=-1)
        preds_dict["ours"].append(ours_pred.item())
        preds_dict["clip"].append(clip_pred.item())
        preds_dict["splice"].append(splice_pred.item())
    return preds_dict

def load_experiment(
    dataset_name: str,
    run_name: str,
    ckpt_path: str | Path,
    enc_path: str | Path,
    split: str = "test",
    device: str = 'cuda',
    target: str = 'ViT-B/32',
    load_baseline: bool = True,
    l1_penalty: float = 0.29,
):
    # check the encoding md5
    check_encoding_md5(run_name, enc_path)

    # load the checkpoint
    model = torch.load(ckpt_path, map_location=device)
    
    # load CLIP model
    clip_model, _ = clip.load(target, device=device)
    clip_model.eval()

    # load the dataset
    if dataset_name == "imagenet":
        load_fn = partial(ImageNetDataset, root=local_config["IMAGENET_ROOT"])
    elif dataset_name == "coco":
        load_fn = partial(CocoDataset, root=local_config["COCO_ROOT"])
    elif dataset_name == "sun397":
        load_fn = partial(SUN397Dataset, root=local_config["DOWNLOAD_ROOT"])
    elif dataset_name == "flowers102":
        load_fn = partial(Flowers102Dataset, root=local_config["DOWNLOAD_ROOT"])
    elif dataset_name == "food101":
        load_fn = partial(Food101Dataset, root=local_config["DOWNLOAD_ROOT"])
    else:   
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = prepare_test(
        load_fn,
        encoding_path=enc_path,
        split=split,
    )
    viz_dataset = load_fn(split=split)

    # load the vocabulary
    with open(enc_path, 'rb') as f:
        data = pickle.load(f)
    vocab = data['vocab']
    assert len(vocab) == dataset[0]['encodings'].shape[0]

    # load the baseline
    if load_baseline:
        import splice
        splicemodel = splice.load(
            name=f'clip:{target}', vocabulary='laion_bigrams', vocabulary_size=15000, device=device,
            l1_penalty=l1_penalty, return_weights=True,
        )
        splice_vocab = splice.get_vocabulary('laion_bigrams', 15000)
    else:
        splicemodel, splice_vocab = None, None

    return (model, vocab), clip_model, (dataset, viz_dataset), (splicemodel, splice_vocab)