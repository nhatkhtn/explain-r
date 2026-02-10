import torch
from torch.utils.data import DataLoader

from exrep.model.surrogate import SurrogateRepresentation

@torch.inference_mode()
def attribute_embeddings(model: SurrogateRepresentation, dataset, device: str, batch_size: int = 1):
    """Compute concept attribution scores for the embeddings of the model."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    W = model.query_encoder.weight              # (output_dim, input_dim)
    B = model.query_encoder.bias                # (output_dim, )
    attributions = []
    for batch in dataloader:
        encodings = batch['encodings'].to(dtype=torch.float32, device=device)       # (batch_size, input_dim)

        repr_unnorm = model.query_encoder(encodings)
        repr_normed = torch.nn.functional.normalize(repr_unnorm, dim=1)             # (batch_size, output_dim)
        λ = 1 / (torch.linalg.vector_norm(repr_unnorm, dim=1) - torch.einsum('d,bd->b', B, repr_normed))
        v = torch.einsum('b,bc,dc,bd->bc', λ, encodings, W, repr_normed)

        attributions.append(v.detach().cpu())
    attributions = torch.cat(attributions, dim=0)                                  # (num_samples, input_dim)
    return attributions.detach().cpu()

def get_top_abs(attributions: torch.Tensor, dim: int, k: int = 5):
    """Get the top-k concepts for each sample."""
    topk_indices = torch.topk(torch.abs(attributions), k, dim=dim).indices
    # reshape the indices to match the original shape
    indices_shape = attributions.shape[:dim] + (k,) + attributions.shape[dim+1:]
    topk_indices = topk_indices.view(indices_shape)
    values = torch.gather(attributions, dim, topk_indices)
    return values, topk_indices

@torch.inference_mode()
def attribute_classes(model: SurrogateRepresentation, encodings: torch.Tensor, texts_embeds: torch.Tensor, device):
    """Compute aggregated cross-modal concept attribution scores for an encoding and text embeddings."""
    W = model.query_encoder.weight              # (output_dim, input_dim)
    B = model.query_encoder.bias                # (output_dim, )

    encodings = encodings.to(dtype=torch.float32, device=device)
    repr_unnorm = model.query_encoder(encodings)
    repr_normed = torch.nn.functional.normalize(repr_unnorm, dim=-1)                # (output_dim)
    repr_nobias = repr_unnorm - B                                                   

    similarity = torch.einsum('d,jd->j', repr_normed, texts_embeds)                 # (num_texts)
    λ = similarity / (torch.einsum('d,jd->j', repr_nobias, texts_embeds))           # (num_texts)

    v = torch.einsum('j,c,dc,jd->jc', λ, encodings, W, texts_embeds)                # (num_texts, input_dim)
    return v.detach().cpu(), similarity.detach().cpu()

@torch.inference_mode()
def attribute_classes_agg(model: SurrogateRepresentation, dataset, text_embeds: torch.Tensor, device):
    """Compute cross-modal concept attribution scores for the classification task."""
    attributions, similarities = [], []
    for datum in dataset:
        encodings = datum['encodings'].to(dtype=torch.float32, device=device)
        v, sim = attribute_classes(model, encodings, text_embeds, device)
        attributions.append(v)
        similarities.append(sim)
    attributions = torch.stack(attributions, dim=0)                                   # (num_inputs, num_texts, input_dim)
    similarities = torch.stack(similarities, dim=0)                                   # (num_inputs, num_texts)
    return attributions.detach().cpu(), similarities