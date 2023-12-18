from src.utils.data_utils import load_dataset
from transformers import AutoProcessor, CLIPSegForImageSegmentation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import spacy
import torch
from tqdm import tqdm


# Experiment configuration
CONFIG = {
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'foil',  # foil/caption
    'segment': 'ambiguous'  # correct/ambiguous/incorrect
}

segment_threshold = 0.25

dataset = load_dataset(**CONFIG)

processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Load English tokenizer, tagger
nlp = spacy.load("en_core_web_sm")

pdf_filename = 'segmentation_' + '_'.join(CONFIG.values()) + '.pdf'
pdf_path = f'../../output/misc/segmentation/{pdf_filename}'

with PdfPages(pdf_path) as pdf:
    for data in tqdm(dataset.values()):

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
            subject = subjects[0]
            inputs = processor(text=[subject], images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            preds = outputs.logits.detach()
            pixels = torch.sigmoid(preds)
            pixels_binary = torch.where(pixels > segment_threshold, 1., 0.)
            pixels_binary_3d = pixels_binary.unsqueeze(-1).expand(*pixels_binary.shape, 3)
            subject_area = pixels.mean()
            subject_area_binary = pixels_binary.mean()

            img_pixels = torch.sigmoid(inputs.pixel_values.squeeze().permute(1, 2, 0))
            img_masked = img_pixels * pixels_binary_3d
            img_masked = torch.where(img_masked == 0., 1., img_masked)

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        [a.axis('off') for a in ax.flatten()]

        ax[0].imshow(image, aspect='auto', extent=[-0.5, 0.5, -0.5, 0.5])  # Align at bottom
        if len(subjects) > 0:
            ax[0].set_title(
                f'{caption}\nSubject: {subject}\nArea (binary): {subject_area:.2f} ({subject_area_binary:.2f})',
                loc='left')
            ax[1].imshow(pixels)
            ax[2].imshow(img_masked)
        else:
            ax[0].set_title(f"Caption: {caption}.\nNo subject found!", loc='left')

        plt.tight_layout()
        # plt.show()

        pdf.savefig(fig)
        plt.close()