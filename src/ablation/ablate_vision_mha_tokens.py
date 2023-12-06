from src.utils.data_utils import load_dataset, generate_clip_input
from src.utils.model_utils import load_model
from src.utils.ablation_utils import create_img_storage_hook_fn, create_ablation_hook_fn, create_token_ablation_hook_fn, store_img_ablation_results

from collections import defaultdict
from pprint import pprint
import torch
from tqdm import tqdm

# Experiment configuration
CONFIG = {
    'encoder': 'vision',    # vision/text
    'component': 'mha_tokens',     # mha/mlp
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'caption',  # foil/caption
    'metric': 'difference', # absolute/difference
    'segment': 'incorrect',  # correct/ambiguous/incorrect
    'effect': 'absolute'    # absolute/relative
}

# Load dataset and model
dataset = load_dataset(**CONFIG)
pprint(CONFIG)
print(f'Input examples: {len(dataset)}')

model, processor = load_model()
ablated_model, _ = load_model()

# Obtain relevant model variables
n_layers = model.vision_model.config.num_hidden_layers
hidden_size = model.vision_model.config.hidden_size
patch_size = model.vision_model.config.patch_size
image_size = model.vision_model.config.image_size
n_positions = (image_size // patch_size) ** 2 + 1 # + cls token

activations = defaultdict(lambda: torch.zeros((n_positions, hidden_size)))

all_hooks = []
for layer_idx, layer in enumerate(model.vision_model.encoder.layers):
    storage_hook_fn = create_img_storage_hook_fn(layer_idx, activations)
    storage_hook = layer.self_attn.register_forward_hook(storage_hook_fn)
    all_hooks.append(storage_hook)

total_score_differences = {'sum': torch.zeros(n_layers, n_positions), 'count': 0}
before_after = defaultdict(lambda: list())

for data in tqdm(dataset.values()):
    inputs = generate_clip_input(data, processor)
    # Get pixel values and calculate average
    img = inputs.pixel_values
    avg_img = img.mean(dim=(-2, -1), keepdim=True).expand_as(img)
    # Create new input objects with averaged pixel values
    new_inputs = inputs.copy()
    new_inputs['pixel_values'] = avg_img
    # Forward pass to record activations with averaged image
    model(**new_inputs)

    # Obtain score of instance without any ablation
    if CONFIG['metric'] == 'absolute':
        score = data['logit_caption']
    if CONFIG['metric'] == 'difference':
        score = data['score']
    # Set up empty tensor to store ablation results for this instance
    score_differences = torch.zeros(n_layers, n_positions)

    # Loop over layers to do ablation
    for l in range(n_layers):
        for p in range(n_positions):
            # Create hook function for this layer
            hook_fn = create_token_ablation_hook_fn(layer=l, position=p, mean_activations=activations)
            # Register hook in this layer of the model
            ablation_hook = ablated_model.vision_model.encoder.layers[l].self_attn.register_forward_hook(hook_fn)
            # Run forward pass to get output with ablation
            output = ablated_model(**inputs)
            if CONFIG['metric'] == 'absolute':
                ablated_score = output.logits_per_text[0].item()
            if CONFIG['metric'] == 'difference':
                ablated_score = (output.logits_per_text[0] - output.logits_per_text[1]).item()
            # Save score difference (normal - ablated) for this layer
            if CONFIG['effect'] == 'absolute':
                score_differences[l, p] = score - ablated_score
            if CONFIG['effect'] == 'relative':
                score_differences[l, p] = ablated_score / score
            # Save original and ablated score to dict
            before_after[(l, p)].append((score, ablated_score))
            # Remove hook
            ablation_hook.remove()
        # break

    # Save results from this example to total results dict
    total_score_differences['sum'] += score_differences
    total_score_differences['count'] += 1
    # break

mean_ablation_effect = (total_score_differences['sum'] / total_score_differences['count']).numpy()

# Remove remaining hooks
for hook in all_hooks:
    hook.remove()

# Save results
store_img_ablation_results(mean_ablation_effect, config=CONFIG)