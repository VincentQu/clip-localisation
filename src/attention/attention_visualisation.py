from src.utils.attention_utils import generate_attention_result_filepath, generate_attention_chart_filepath

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = {
    'encoder': 'text',
    'negation': 'foil',
    'segment': None
}

h5_filepath = generate_attention_result_filepath(CONFIG)

with h5py.File(h5_filepath, 'r') as hdf:
    # Load results
    max_attn_diff_results = np.array(hdf['results'])
    attn_correlations = np.array(hdf['correlations'])


OUTPUT_FORMAT = 'eps' # [eps, png]

x_tick_labels = [i + 1 for i in range(max_attn_diff_results.shape[1])]
y_tick_labels = [i + 1 for i in range(max_attn_diff_results.shape[0])]

if OUTPUT_FORMAT == 'png':
    fig, ax = plt.subplots(2, 1, figsize=(9, 9))

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
    plot_filepath = generate_attention_chart_filepath(CONFIG)
    plt.show()
    # plt.savefig(plot_filepath)
    print(f'Plot saved at {plot_filepath}')

if OUTPUT_FORMAT == 'eps':

    avg_attn_diff_per_layer = max_attn_diff_results.mean(axis=0)

    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 11,
        "font.family": "Computer Modern Roman"
    })

    fig, axes = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1],
                                                'width_ratios': [15, 1]})

    sns.set_theme(style="whitegrid")
    attn_hm = sns.heatmap(max_attn_diff_results, annot=True, fmt='.2f', vmin=0, vmax=0.6, ax=axes[0,0], cbar_ax=axes[0,1])
    attn_hm.set(ylabel='Head', yticklabels=y_tick_labels)

    attn_hm.set_xticks([])

    bp = sns.barplot(avg_attn_diff_per_layer, ax=axes[1, 0], color='#02456b', linewidth=0)
    bp.set(xlabel='Layer', ylabel='Average', xticklabels=x_tick_labels)

    axes[1, 1].axis('off')

    plt.tight_layout(pad=0.8)
    plot_filepath_eps = generate_attention_chart_filepath(CONFIG).replace('.png', '.eps')
    plt.savefig(plot_filepath_eps, format='eps')
    print(f'Plot saved at {plot_filepath_eps}')
