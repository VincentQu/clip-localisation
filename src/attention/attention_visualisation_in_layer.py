from src.utils.attention_utils import generate_attention_result_filepath, generate_attention_chart_filepath

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = {
    'encoder': 'text',
    'negation': 'foil',
    'segment': None,
    'layer_of_interest': 3
}

h5_filepath = generate_attention_result_filepath(CONFIG)

with h5py.File(h5_filepath, 'r') as hdf:
    # Load results
    avg_attention_diff = np.array(hdf['results'])


attn_head_labels = [i + 1 for i in range(avg_attention_diff.shape[0])]

OUTPUT_FORMAT = 'png' # [eps, png]

generic_token_labels = ['[SOT]',
                        'there',
                        'is/are',
                        'a/some',
                        'first subject token',
                        'further subject tokens',
                        '.',
                        '[EOT]']


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

    plt.show()
    # plt.savefig(plot_filepath)

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