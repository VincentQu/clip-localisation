import json
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

def load_dataset(type='rephrased', min_score=-1000, min_logit_caption=-1000, min_logit_foil=-1000, **kwargs):
    """
    If no arguments are provided this will return correctly classified examples from the rephrased dataset where the negation is in the foil
    """

    # Merge default filter values with those provided via kwargs
    default_filters = {'negation': 'foil', 'correct': True}
    category_filters = {**default_filters, **kwargs}

    valid_types = ['standard', 'rephrased']
    assert type in valid_types, f'type argument must be one of {valid_types}'

    dataset_path = f'../../data/datasets/valse/existence_{type}.json'
    with open(dataset_path, 'rb') as f:
        valse_existence = json.load(f)

    # Filter based on categories (e.g., negation)
    dataset_filtered = {k: v for k, v in valse_existence.items() if all(v.get(key) == val for key, val in category_filters.items() if val is not None and key in v.keys())}
    # Filter based on min values
    dataset_filtered = {k: v for k, v in dataset_filtered.items() if v.get('score') > min_score and v.get('logit_caption') > min_logit_caption and v.get('logit_foil') > min_logit_foil}
    # Return filtered dataset as dict
    return dataset_filtered