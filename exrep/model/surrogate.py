import torch


class SurrogateRepresentation(torch.nn.Module):
    def __init__(self, local_dim: int, repr_dim: int, output_dim: int, temperature: float, use_key_encoder=False, device: str = "cuda"):
        super().__init__()
        self.temperature = temperature
        self.query_encoder = torch.nn.Linear(local_dim, output_dim, device=device)
        if use_key_encoder:
            self.key_encoder = torch.nn.Linear(repr_dim, output_dim, device=device)
        else:
            self.key_encoder = torch.nn.Identity()
        # self.scale = torch.nn.Parameter(torch.ones(1, local_dim, device=device))

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(x)
        # return self.query_encoder(x * self.scale)
    
    def encode_key(self, x: torch.Tensor) -> torch.Tensor:
        return self.key_encoder(x)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor):
        return self.encode_query(x_q), self.encode_key(x_k)
    
    @staticmethod
    def initialize(*, device : str = 'cuda', **kwargs):
        if kwargs.get('mlp_dim', None) is not None:
            return MLPSurrogate(device=device, **kwargs)
        else:
            return SurrogateRepresentation(device=device, **kwargs)
    
LocalRepresentationApproximator = SurrogateRepresentation

class MLPSurrogate(SurrogateRepresentation):
    def __init__(self, local_dim: int, output_dim: int, mlp_dim: int, device: str = "cuda", **kwargs):
        super().__init__(local_dim=local_dim, output_dim=output_dim, device=device, **kwargs)
        self.mlp_encoder = torch.nn.Sequential(
            torch.nn.Linear(local_dim, mlp_dim, device=device),
            torch.nn.Sigmoid(),
            torch.nn.Linear(mlp_dim, output_dim, device=device),
        )
        
    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(x) + self.mlp_encoder(x)