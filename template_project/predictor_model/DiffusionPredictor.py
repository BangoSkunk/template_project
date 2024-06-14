import json

import torch
from diffusers import StableDiffusionPipeline

from .PredictorInterface import PredictorInterface

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class DiffusionPredictor(PredictorInterface):
    def __init__(self, model_config: str):
        model_config_dict = json.loads(model_config)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_config_dict.get("model_id"),
        ).to(device)
        self.num_inference_steps = model_config_dict.get("num_inference_steps")

    def predict(self, data: dict) -> list:
        prompt = data["prompt"]
        prediction = self.pipe(
            prompt,
            num_inference_steps=self.num_inference_steps,
        ).images
        return prediction
