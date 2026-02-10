# from .sam2_batch import SAM2BatchMaskGenerator

def init_target(name: str, **kwargs):
    """Initialize the model to be explained."""
    if name == 'mocov3':
        from .mocov3 import MoCoV3
        model = MoCoV3.from_pretrained('r-50-100ep.pth.tar', **kwargs)
        # this is not necessary if we use the right context manager
        # but it is for good measure
        for param in model.parameters():
            param.requires_grad = False

    elif name == 'clip':
        from .clip_model import CLIP
        model = CLIP(**kwargs)
        
    else:
        raise ValueError(f"Unknown target model {name}")
    
    model.eval()
    return model