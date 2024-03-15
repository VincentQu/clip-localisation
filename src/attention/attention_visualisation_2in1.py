from src.utils.attention_utils import generate_attention_result_filepath, generate_attention_chart_filepath

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

CONFIG_FOIL = {
    'encoder': 'text',
    'negation': 'foil',
    'segment': None
}

h5_filepath_foil = generate_attention_result_filepath(CONFIG_FOIL)

with h5py.File(h5_filepath_foil, 'r') as hdf:
    # Load results
    results_foil = np.array(hdf['results'])

results_foil_annot = np.array([[f'{x:.2f}'.lstrip('0').replace('-0', '-') for x in row] for row in results_foil])
avg_results_foil = results_foil.mean(axis=0)

CONFIG_CAPTION = {
    'encoder': 'text',
    'negation': 'caption',
    'segment': None
}

h5_filepath_caption = generate_attention_result_filepath(CONFIG_CAPTION)

with h5py.File(h5_filepath_caption, 'r') as hdf:
    # Load results
    results_caption = np.array(hdf['results'])

results_caption_annot = np.array([[f'{x:.2f}'.lstrip('0').replace('-0', '-') for x in row] for row in results_caption])
avg_results_caption = results_caption.mean(axis=0)

x_tick_labels = [i + 1 for i in range(results_foil.shape[1])]
y_tick_labels = [i + 1 for i in range(results_foil.shape[0])]

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 9
})

fig, axes = plt.subplots(2, 3, figsize=(6.4,3.5), gridspec_kw={'height_ratios': [3, 1],
                                            'width_ratios': [20,20,1]})

sns.set_theme(style="whitegrid")

attn_hm_foil = sns.heatmap(results_foil, annot=results_foil_annot, fmt='', annot_kws={'fontsize':8.0}, vmin=0, vmax=0.6, ax=axes[0,0], cbar_ax=axes[0,2])
attn_hm_foil.set(ylabel='Attention head', yticklabels=y_tick_labels)
attn_hm_foil.set_title('Negation in foil', size=10)
attn_hm_foil.set_xticks([])

bp_foil = sns.barplot(avg_results_foil, ax=axes[1, 0], color='#02456b', linewidth=0)
bp_foil.set(xlabel='Layer', ylabel='Average', xticklabels=x_tick_labels)
axes[1,0].set(ylim=[0,0.1])

attn_hm_caption = sns.heatmap(results_caption, annot=results_caption_annot, fmt='', annot_kws={'fontsize':8.0}, vmin=0, vmax=0.6, ax=axes[0,1], cbar=False)
attn_hm_caption.set(yticklabels=y_tick_labels)
attn_hm_caption.set_title('Negation in caption', size=10)
attn_hm_caption.set_xticks([])
attn_hm_caption.set_yticks([])

bp_caption = sns.barplot(avg_results_caption, ax=axes[1, 1], color='#02456b', linewidth=0)
bp_caption.set(xlabel='Layer', xticklabels=x_tick_labels)
bp_caption.set_yticks([])
axes[1,1].set(ylim=[0,0.1])

axes[1, 2].axis('off')

plt.tight_layout(pad=0.5)

# plt.show()

plot_filepath_eps = '../../output/charts/attention/attention_2in1.eps'
plt.savefig(plot_filepath_eps, format='eps')
print(f'Plot saved at {plot_filepath_eps}')