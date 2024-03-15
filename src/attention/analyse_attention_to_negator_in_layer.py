from src.utils.model_utils import load_model
from src.utils.data_utils import load_dataset, generate_clip_input
from src.utils.attention_utils import store_attention_results

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

CONFIG = {
    'encoder': 'text',
    'negation': 'caption',
    'segment': None,
    'layer_of_interest': 3
}

OUTPUT_FORMAT = 'eps' # [eps, png]

model, processor = load_model()
layers = model.text_model.config.num_hidden_layers
attn_heads = model.text_model.config.num_attention_heads

# Extract vocabulary
stoi = processor.tokenizer.vocab
itos = {i: s for s, i in stoi.items()}
# Determine index of "no" token
no_token = stoi['no</w>']
no_position = 3 # SOT, There, is/are, no

dataset = load_dataset(**CONFIG)

# Determine which input positions should be aggregated, to be able to compare inputs of different length
position_agg_start = 5 # SOT, There, is/are, no, subject, ...
position_agg_end = -2 # dot, EOT
positions_analysed = position_agg_start - position_agg_end + 1

# Loop over data
attentions_list = []
attentions_diff_list = []

for data in tqdm(dataset.values()):
    # Generate input
    inputs = generate_clip_input(data, processor)
    negation_idx = 1 if data['negation'] == 'foil' else 0

    # Forward pass
    output = model(**inputs, output_attentions=True)

    # Extract attention to "no" from forward pass
    attention_to_no = output.text_model_output.attentions[CONFIG['layer_of_interest']][negation_idx, :, :,
                      no_position].detach().numpy()  # caption/foil, head, row, col

    attention_to_some = output.text_model_output.attentions[CONFIG['layer_of_interest']][abs(negation_idx - 1), :, :,
                      no_position].detach().numpy()  # caption/foil, head, row, col

    attention_diff = attention_to_no - attention_to_some

    # 1. ABSOLUTE
    # Set up new array to hold attention aggregated over positions
    attention_to_no_agg = np.zeros((attn_heads, positions_analysed))
    attention_to_no_agg[:, :position_agg_start] = attention_to_no[:, :position_agg_start]
    # Further subject positions (aggregated)
    further_positions = attention_to_no[:, position_agg_start:position_agg_end].shape[1]
    if further_positions > 0:
        attention_to_no_agg[:, position_agg_start] = attention_to_no[:, position_agg_start:position_agg_end].max(axis=1)
    attention_to_no_agg[:, position_agg_end:] = attention_to_no[:, position_agg_end:]

    # Store result in list
    attentions_list.append(attention_to_no_agg)
    # break

    # 2. DIFFERENCE (no - some)
    # Set up new array to hold attention difference aggregated over positions
    attention_diff_agg = np.zeros((attn_heads, positions_analysed))
    attention_diff_agg[:, :position_agg_start] = attention_diff[:, :position_agg_start]
    # Further subject positions (aggregated)
    further_positions = attention_diff[:, position_agg_start:position_agg_end].shape[1]
    if further_positions > 0:
        attention_diff_agg[:, position_agg_start] = attention_diff[:, position_agg_start:position_agg_end].max(axis=1)
    attention_diff_agg[:, position_agg_end:] = attention_diff[:, position_agg_end:]

    # Store result in list
    attentions_diff_list.append(attention_diff_agg)

# Stack and average results over dataset
avg_attention = np.stack(attentions_list).mean(axis=0)
avg_attention_diff = np.stack(attentions_diff_list).mean(axis=0)

store_attention_results(avg_attention_diff, correlations=None, config=CONFIG)

"""# Plotting
generic_token_labels = ['[SOT]',
                        'there',
                        'is/are',
                        'a/some',
                        'first subject token',
                        'further subject tokens',
                        '.',
                        '[EOT]']

attn_head_labels = [i + 1 for i in range(attn_heads)]

if OUTPUT_FORMAT == 'png':

    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    sns.heatmap(avg_attention_diff.transpose(), annot=True, fmt='.2f', ax=ax,
                xticklabels=attn_head_labels, yticklabels=generic_token_labels)
    ax.set(xlabel="Attention head", ylabel="Position")
    plt.suptitle("How much does each position attend to the 'no' token?", fontsize=16, fontweight='bold')
    ax.set_title(f"Layer {CONFIG['layer_of_interest'] + 1} - Negation in the {CONFIG['negation']}", fontsize=14)
    # plt.show()

    plt.tight_layout()

    plot_filename = f"attention_layer_{'_'.join([str(v) for v in CONFIG.values()])}.png"
    plot_filepath = f"../../output/charts/attention/{plot_filename}"

    plt.savefig(plot_filepath)

if OUTPUT_FORMAT == 'eps':

    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 11
    })

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(avg_attention_diff.transpose(), annot=True, fmt='.2f', vmin=0, vmax=0.6, ax=ax,
                xticklabels=attn_head_labels, yticklabels=generic_token_labels)
    ax.set(xlabel="Attention head", ylabel="Position")
    ax.set_title(f"Layer {CONFIG['layer_of_interest'] + 1}")

    plt.tight_layout()

    plot_filename_eps = f"attention_layer_{'_'.join([str(v) for v in CONFIG.values()])}.eps"
    plot_filepath = f"../../output/charts/attention/{plot_filename_eps}"

    plt.savefig(plot_filepath, format='eps')
    print(f'Output saved at {plot_filepath}')
"""