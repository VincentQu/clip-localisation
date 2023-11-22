from src.utils.data_utils import load_dataset, generate_clip_input
from src.utils.model_utils import load_model
from src.utils.ablation_utils import create_storage_hook_fn, create_ablation_hook_fn, store_ablation_results


from collections import defaultdict
import torch
from tqdm import tqdm

# Experiment configuration
CONFIG = {
    'encoder': 'vision',    # vision/text
    'component': 'mha',     # mha/mlp
    'dataset': 'rephrased', # standard/rephrased
    'correct': True,        # True/False
    'negation': 'foil'      # foil/caption
}

# Load dataset and model
dataset = load_dataset(**CONFIG)
print(f'Input examples: {len(dataset)}')
model, processor = load_model()

# Obtain relevant model variables
n_layers = model.vision_model.config.num_hidden_layers
hidden_size = model.vision_model.config.hidden_size
patch_size = model.vision_model.config.patch_size
image_size = model.vision_model.config.image_size
tokens = (image_size // patch_size) ** 2 + 1 # + cls token

# Calculate mean activations across dataset

activations = defaultdict(lambda: {'sum': torch.zeros((tokens, hidden_size)), 'count': 0})

all_hooks = []
for layer_idx, layer in enumerate(model.vision_model.encoder.layers):
    storage_hook_fn = create_storage_hook_fn(layer_idx, activations)
    storage_hook = layer.self_attn.register_forward_hook(storage_hook_fn)
    all_hooks.append(storage_hook)

print('Recording activations in dataset')
for data in tqdm(dataset.values()):
    inputs = generate_clip_input(data, processor)
    model(**inputs)

for hook in all_hooks:
    hook.remove()

avg_activations = {layer: data['sum'] / data['count'] for layer, data in activations.items()}

# Ablate MHA outputs

total_score_differences = {'sum': torch.zeros(n_layers), 'count': 0}
# Loop over data
print(f"Running ablation of {CONFIG['component'].upper()} in {CONFIG['encoder']} encoder")
for data in tqdm(dataset.values()):
    # Generate model inputs
    inputs = generate_clip_input(data, processor)
    # Obtain score (in this case caption logit) of example without any ablation
    score = data['logit_caption']
    # Set up empty tensor to store ablation results for this example
    score_differences = torch.zeros(n_layers)
    # Loop over layers to ablate in each layer separately
    for l in range(n_layers):
        # Create hook function for this layer
        hook_fn = create_ablation_hook_fn(layer=l, mean_activations=avg_activations)
        # Register hook in this layer of the model
        ablation_hook = model.vision_model.encoder.layers[l].self_attn.register_forward_hook(hook_fn)
        # Run forward pass to get output with ablation
        output = model(**inputs)
        ablated_score = output.logits_per_text[0].item()
        # Save ablation result and remove hook
        score_differences[l] = score - ablated_score
        ablation_hook.remove()
    # Save results from this example to total results dict
    total_score_differences['sum'] += score_differences
    total_score_differences['count'] += 1

mean_ablation_effect = (total_score_differences['sum'] / total_score_differences['count']).numpy()

# Create dictionaries to save results
activation_dict = {layer: val.cpu().detach().numpy() for layer, val in avg_activations.items()}
effect_dict = {layer: val for layer, val in enumerate(mean_ablation_effect)}

# Save results
store_ablation_results(activation_dict, effect_dict, CONFIG)