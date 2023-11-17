from PIL import Image
from transformers import CLIPProcessor
from typing import Dict, Any
def generate_clip_input(data: Dict[str, Any], processor: CLIPProcessor):
    # Get image and text (caption + foil)
    image = Image.open(f"../../data/raw/visual7w/images/{data['image_file']}")
    text = [data['caption'], data['foil']]

    # Create input object with CLIP processor
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

    return inputs
