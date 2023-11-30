import pandas as pd

from src.utils.model_utils import load_model
from src.utils.data_utils import load_dataset, generate_clip_input
from src.utils.attention_utils import generate_ablation_chart_filepath

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

CONFIG = {
    'negation': 'caption',
    'segment': None
}

model, processor = load_model()
layers = model.text_model.config.num_hidden_layers
attn_heads = model.text_model.config.num_attention_heads

# Extract vocabulary
stoi = processor.tokenizer.vocab
itos = {i: s for s, i in stoi.items()}
# Determine index of "no" token
no_token = stoi['no</w>']

dataset = load_dataset(**CONFIG)

max_attn_diffs = []
rows = []

for data in tqdm(dataset.values()):
    inputs = generate_clip_input(data, processor)

    # Get the position of the no token in the caption or foil
    if CONFIG['negation'] == 'foil':
        no_token_position = (inputs.input_ids[1] == no_token).nonzero().item()
    if CONFIG['negation'] == 'caption':
        no_token_position = (inputs.input_ids[0] == no_token).nonzero().item()

    output = model(**inputs, output_attentions=True)

    # Extract attention from caption and foil forward pass
    attentions = output.text_model_output.attentions
    att_c = torch.stack([attentions[i][0] for i in range(layers)])  # layer, head, token, token
    att_f = torch.stack([attentions[i][1] for i in range(layers)])  # layer, head, token, token

    max_attn_diff = np.zeros((attn_heads, layers))
    for l in range(layers):
        for h in range(attn_heads):
            if CONFIG['negation'] == 'foil':
                attention_to_no = att_f[l, h, no_token_position + 1:, no_token_position]
                attention_to_some = att_c[l, h, no_token_position + 1:, no_token_position]
            if CONFIG['negation'] == 'caption':
                attention_to_no = att_c[l, h, no_token_position + 1:, no_token_position]
                attention_to_some = att_f[l, h, no_token_position + 1:, no_token_position]
            attention_diff = attention_to_no - attention_to_some
            max_attn_diff[h, l] = attention_diff.max().item()

            row = (h, l, data['score'], attention_diff.max().item())
            rows.append(row)

    max_attn_diffs.append(max_attn_diff)

attn_results = pd.DataFrame(data=rows, columns=['Head', 'Layer', 'Score', 'Max Attn Diff'])
attn_correlations = attn_results.groupby(['Head', 'Layer']).\
    apply(lambda x: x['Score'].corr(x['Max Attn Diff'])).\
    rename('Correlation').\
    reset_index().\
    sort_values(by='Correlation', ascending=False).\
    pivot(index="Head", columns="Layer", values="Correlation")

max_attn_diff_results = np.stack(max_attn_diffs).mean(0)


fig, ax = plt.subplots(2, 1, figsize=(9, 9))  # Adjust the figsize as needed

x_tick_labels = [i + 1 for i in range(layers)]
y_tick_labels = [i + 1 for i in range(attn_heads)]

attn_hm = sns.heatmap(max_attn_diff_results, annot=True, fmt='.2f', vmin=0, vmax=0.6, ax=ax[0])
attn_hm.set(xlabel='Layer', ylabel='Head', xticklabels=x_tick_labels, yticklabels=y_tick_labels,
            title='Difference in max attention to "no" vs. "some/a(n)" token')

# Add text annotations
for idx, (key, value) in enumerate(CONFIG.items()):
    ax[0].text(1.2, 1 - idx * 0.07, f'{str(key).capitalize()}: {str(value).upper()}',
                 transform=ax[0].transAxes, verticalalignment='top')

corr_hm = sns.heatmap(attn_correlations, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap='vlag', ax=ax[1])
corr_hm.set(xlabel="Layer", ylabel="Head", xticklabels=x_tick_labels, yticklabels=y_tick_labels,
            title='Correlation between attn diff (s.a.) and CLIP caption/foil classification performance')

plt.tight_layout()
plot_filepath = generate_ablation_chart_filepath(CONFIG)
# plt.show()
plt.savefig(plot_filepath)
print(f'Plot saved at {plot_filepath}')
