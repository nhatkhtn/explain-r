import torch

from exrep.model.surrogate import SurrogateRepresentation

class DistilledRepresentationModel(torch.nn.Module):
    def __init__(self, target_model: torch.nn.Module, surrogate_model: SurrogateRepresentation):
        super().__init__()
        self.target_model = target_model
        self.surrogate_model = surrogate_model

    def forward(self, images: torch.Tensor, texts: torch.Tensor, encodings: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
        # compute teacher embeddings
        # we can't use inferece mode here, because we need to compute the gradients
        # for the surrogate model
        with torch.no_grad():
            # in CLIP, we call the images 'queries' and the texts 'keys'
            teacher_queries_embeds = self.target_model.embed_query(images)
            teacher_queries_projected = self.target_model.project_query(teacher_queries_embeds)
            teacher_keys_embeds = self.target_model.embed_key(texts)
            teacher_keys_projected = self.target_model.project_key(teacher_keys_embeds)

        # compute student embeddings
        student_queries_embeds, student_keys_embeds = self.surrogate_model(encodings, teacher_keys_embeds)

        return student_queries_embeds, student_keys_embeds, teacher_queries_projected, teacher_keys_projected