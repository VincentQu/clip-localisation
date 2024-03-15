from src.utils.attention_utils import generate_attention_result_filepath, generate_attention_chart_filepath

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

CONFIG_FOIL = {
    'encoder': 'text',
    'negation': 'foil',
    'segment': None,
    'layer_of_interest': 3
}

h5_filepath_foil = generate_attention_result_filepath(CONFIG_FOIL)

with h5py.File(h5_filepath_foil, 'r') as hdf:
    # Load results
    results_foil = np.array(hdf['results'])

results_foil_annot = np.array([[f'{x:.2f}'.lstrip('0').replace('-0', '-') for x in row] for row in results_foil]).transpose()

CONFIG_CAPTION = {
    'encoder': 'text',
    'negation': 'caption',
    'segment': None,
    'layer_of_interest': 3
}

h5_filepath_caption = generate_attention_result_filepath(CONFIG_CAPTION)

with h5py.File(h5_filepath_caption, 'r') as hdf:
    # Load results
    results_caption = np.array(hdf['results'])

results_caption_annot = np.array([[f'{x:.2f}'.lstrip('0').replace('-0', '-') for x in row] for row in results_caption]).transpose()

attn_head_labels = [i + 1 for i in range(results_caption.shape[0])]

OUTPUT_FORMAT = 'png' # [eps, png]

generic_token_labels = ['[SOT]',
                        'there',
                        'is/are',
                        'a/some',
                        r'\textit{First subject token}',
                        r'\textit{Further subject tokens}',
                        '.',
                        '[EOT]']


plt.rcParams.update({
    "text.usetex": True,
    "font.size": 10
})

fig, axes = plt.subplots(1, 3, figsize=(6.4,3), gridspec_kw={'width_ratios': [20,20,1]})

sns.set_theme(style="whitegrid")

attn_hm_foil = sns.heatmap(results_foil.transpose(), annot=results_foil_annot, fmt='', annot_kws={'fontsize':8.0},
                           vmin=0, vmax=0.6, ax=axes[0], cbar_ax=axes[2],
                           xticklabels=attn_head_labels, yticklabels=generic_token_labels)
attn_hm_foil.set(xlabel="Attention head")
attn_hm_foil.set_title("Negation in foil", size=11)

attn_hm_caption = sns.heatmap(results_caption.transpose(), annot=results_caption_annot, fmt='', annot_kws={'fontsize':8.0},
                              vmin=0, vmax=0.6, ax=axes[1], cbar_ax=axes[2],
                              xticklabels=attn_head_labels, yticklabels=generic_token_labels)
attn_hm_caption.set(xlabel="Attention head")
attn_hm_caption.set_title("Negation in caption", size=11)
attn_hm_caption.set_yticks([])

plt.tight_layout(pad=0.5)
# plt.show()

plot_filename_eps = f"attention_layer_{CONFIG_FOIL['layer_of_interest']}_2in1.eps"
plot_filepath = f"../../output/charts/attention/{plot_filename_eps}"

plt.savefig(plot_filepath, format='eps')
print(f'Output saved at {plot_filepath}')
