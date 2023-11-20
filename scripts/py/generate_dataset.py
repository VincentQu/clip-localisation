import json
from tqdm import tqdm
import os

from src.utils.model_utils import load_model
from src.utils.data_utils import generate_clip_input

# Open raw valse json
with open('../../data/raw/valse/valse_existence.json', 'rb') as f:
    valse_existence = json.load(f)

model, processor = load_model()

dataset = dict()

for key, data in tqdm(valse_existence.items()):
    if data['mturk']['caption'] >= 2:
        inputs = generate_clip_input(data, processor)
        outputs = model(**inputs, return_dict=True)
        logit_caption, logit_foil = outputs.logits_per_text.squeeze().tolist()
        entry = {
            'dataset_idx': data['dataset_idx'],
            'caption': data['caption'],
            'foil': data['foil'],
            'image_file': data['image_file'],
            'negation': 'caption' if data['provenance_of_foils'] == 'zero_to_something' else 'foil',
            'logit_caption': logit_caption,
            'logit_foil': logit_foil,
            'score': logit_caption - logit_foil,
            'correct': logit_caption > logit_foil
        }
        dataset[key] = entry

with open('../../data/datasets/valse/existence.json', 'w') as f:
    json.dump(dataset, f, indent=4, sort_keys=True)