import logging
from typing import Any

import torch
from transformers.tokenization_utils_base import BatchEncoding

TensorDict = dict[str, torch.Tensor] | BatchEncoding

logger = logging.getLogger(__name__)

def torch_pairwise_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise cosine similarity between two tensors.

    Args:
        x (torch.Tensor): Tensor of shape (n_samples, n_features).
        y (torch.Tensor): Tensor of shape (n_samples, n_features).

    Returns:
        torch.Tensor: Tensor of shape (n_samples, n_samples) containing the pairwise cosine similarity.
    """
    return torch.nn.functional.cosine_similarity(x[:,:,None], y.t()[None,:,:])  

def pythonize(d: dict[Any, torch.Tensor]) -> dict[str, Any]:
    """Convert a dictionary of torch single-element tensors to a dictionary of python objects.
    
    Mostly used for logging purposes."""
    return {key: value.item() for key, value in d.items()}

class Nop:
    def nop(*args, **kw): pass
    def __getattr__(self, _): return self.nop
