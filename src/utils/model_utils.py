from transformers import CLIPModel, CLIPProcessor
from typing import Tuple


def load_model(version: str = 'openai/clip-vit-base-patch32') -> Tuple[CLIPModel, CLIPProcessor]:
    # Load model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Extract model components
    # text_model = model.text_model
    # vision_model = model.vision_model

    # Load processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor
