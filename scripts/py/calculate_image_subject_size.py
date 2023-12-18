from copy import deepcopy
import json
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation
import spacy
from tqdm import tqdm

# Load model and processor
processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Load English tokenizer
nlp = spacy.load("en_core_web_sm")

# Set activation threshold for binary pixel map
segment_threshold = 0.25

# Open rephrased VALSE existence
with open('../../data/datasets/valse/existence_rephrased.json', 'r') as f:
    existence_rephrased = json.load(f)


new_dataset = deepcopy(existence_rephrased)
for key, data in tqdm(existence_rephrased.items()):
    # print(data)

    # Load image (and expand to 3D if 2D)
    image = Image.open(f"../../data/raw/visual7w/images/{data['image_file']}")
    if np.array(image).ndim == 2:
        _img = np.array(image)
        image = Image.fromarray(np.stack([_img] * 3, axis=-1))

    # Try to extract subject from caption
    caption = data['caption']
    doc = nlp(caption)
    subjects = [c.text for c in doc.noun_chunks]
    # If subject was found, run CLIPSeg on image using the subject as the object to find in the image
    if len(subjects) > 0:
        # Forward pass through CLIPSeg
        subject = subjects[0]
        inputs = processor(text=[subject], images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        # Get predictions and convert to pixel values
        preds = outputs.logits.detach()
        pixels = torch.sigmoid(preds)
        pixels_binary = torch.where(pixels > segment_threshold, 1., 0.)
        # Calculate relative size of identified subject in image
        subject_area = pixels.mean().item()
        subject_area_binary = pixels_binary.mean().item()
        new_dataset[key]['subject'] = subject
        new_dataset[key]['subject_area'] = subject_area
        new_dataset[key]['subject_area_binary'] = subject_area_binary
    else:
        new_dataset[key]['subject'] = None
        new_dataset[key]['subject_area'] = None
        new_dataset[key]['subject_area_binary'] = None

with open('../../data/datasets/valse/existence_rephrased.json', 'w') as f:
    json.dump(new_dataset, f, indent=4, sort_keys=True)