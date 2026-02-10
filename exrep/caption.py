import itertools

from transformers import AutoProcessor, Blip2ForConditionalGeneration, AddedToken
import torch

class CaptioningModel(torch.nn.Module):
    """Light wrapper around the BLIP2 model for captioning."""
    def __init__(self, model="Salesforce/blip2-opt-2.7b-coco", device='cuda'):
        super().__init__()
        self.model = Blip2ForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16).to(device)
        self.processor = AutoProcessor.from_pretrained(model, use_fast=True)
        self.model.eval()

        # patch the blip model 
        # as in https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042
        self.processor.num_query_tokens = self.model.config.num_query_tokens
        image_token = AddedToken("<image>", normalized=False, special=True)
        self.processor.tokenizer.add_tokens([image_token], special_tokens=True)
        self.model.resize_token_embeddings(len(self.processor.tokenizer), pad_to_multiple_of=64) # pad for efficient computation
        self.model.config.image_token_index = len(self.processor.tokenizer) - 1

        # not sure if they actually make the model faster
        # model.generation_config.cache_implementation = "static"

        # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    @torch.inference_mode()
    def forward(self, inputs, max_new_tokens=100, num_captions_per_image=5, do_sample=True, top_p=0.9, **kwargs):
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_captions_per_image,
            do_sample=do_sample,    # nucleus sampling + multinomial sampling
            top_p=top_p,
            **kwargs,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = map(str.strip, generated_text)
        return list(itertools.batched(generated_text, num_captions_per_image))


