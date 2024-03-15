import json
from tqdm import tqdm
import torch
import os

from src.utils.model_utils import load_model
from src.utils.data_utils import generate_clip_input

segment_thresholds = [-1, 1]

# Open raw valse json
with open('../../data/raw/valse/valse_existence.json', 'rb') as f:
    valse_existence = json.load(f)

# Rephrase inputs
valse_existence_rephrased = dict()

print('Rephrasing inputs')

for key, data in tqdm(valse_existence.items()):
    new_data = data.copy()
    caption, foil = [new_data['caption'], new_data['foil']]

    if caption.startswith('There are'):

        if new_data['provenance_of_foils'] == 'something_to_zero':
            # Modify caption
            caption_words = caption.split(' ')
            are_idx = caption_words.index('are')
            caption_words.insert(are_idx + 1, 'some')
            caption_new = ' '.join(caption_words)
            new_data['caption'] = caption_new

        if new_data['provenance_of_foils'] == 'zero_to_something':
            # Modify foil
            foil_words = foil.split(' ')
            are_idx = foil_words.index('are')
            foil_words.insert(are_idx + 1, 'some')
            foil_new = ' '.join(foil_words)
            new_data['foil'] = foil_new

        # Add item to new dict
        valse_existence_rephrased[key] = new_data

    if caption.startswith('There is'):
        # Add item without modification
        valse_existence_rephrased[key] = new_data

# Run rephrased inputs through CLIP to generate dataset

model, processor = load_model()

dataset = dict()

print('Creating dataset from rephrased input')
for key, data in tqdm(valse_existence_rephrased.items()):
    if data['mturk']['caption'] >= 2:
        inputs = generate_clip_input(data, processor)
        outputs = model(**inputs, return_dict=True)
        logit_caption, logit_foil = outputs.logits_per_text.squeeze().tolist()
        score = logit_caption - logit_foil
        caption_embed, foil_embed = outputs.text_embeds
        caption_foil_dot = torch.dot(caption_embed, foil_embed).item()
        entry = {
            'dataset_idx': data['dataset_idx'],
            'caption': data['caption'],
            'foil': data['foil'],
            'image_file': data['image_file'],
            'negation': 'caption' if data['provenance_of_foils'] == 'zero_to_something' else 'foil',
            'logit_caption': logit_caption,
            'logit_foil': logit_foil,
            'similarity_caption_foil': caption_foil_dot,
            'positions': inputs.input_ids.shape[1],
            'score': score,
            'correct': score > 0,
            'segment': 'incorrect' if score < segment_thresholds[0] else 'ambiguous' if score < segment_thresholds[1] else 'correct'
        }
        dataset[key] = entry

with open('../../data/datasets/valse/existence_rephrased.json', 'w') as f:
    json.dump(dataset, f, indent=4, sort_keys=True)