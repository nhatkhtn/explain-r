import torch
import torch.nn as nn
import clip

class CLIP(nn.Module):
    def __init__(self, variant='RN50', **kwargs):
        super().__init__()
        self.model, _ = clip.load(variant, **kwargs)

    def embed_query(self, x: torch.Tensor):
        """The image is called query in CLIP
        
        Args:
            x: transformed image tensor with shape (batch_size, C, H, W)
        """
        return self.model.encode_image(x).float()

    def embed_key(self, x: torch.Tensor):
        """The text is called key in CLIP.

        Args:
            x: tokenized text
        """
        return self.model.encode_text(x).float()

    def project_query(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def project_key(self, x: torch.Tensor) -> torch.Tensor:
        return x