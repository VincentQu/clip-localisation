from src.utils.ablation_utils import generate_ablation_result_filepath, generate_ablation_chart_filepath

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# Experiment configuration
CONFIG = {
    'encoder': 'vision',    # vision/text
    'component': 'mha_tokens',     # mha/mlp
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'foil',  # foil/caption
    'metric': 'difference', # absolute/difference
    'segment': 'incorrect',  # correct/ambiguous/incorrect
    'effect': 'absolute'    # absolute/relative
}


h5_filepath = generate_ablation_result_filepath(CONFIG)

try:
    with h5py.File(h5_filepath, 'r') as hdf:
        # Load effects
        effects = np.array(hdf['effects'])
        # Load before after
        before_after = np.array(hdf['before_after'])
except:
    pass

if 'token' in CONFIG['component']:
    effect_df = pd.DataFrame(effects).transpose()
    effect_df_annot = effect_df.map(lambda x: f'{x:.2f}' if x >= 0.02 or x <=-0.02 else '')

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    x_tick_labels = [i + 1 for i in range(effect_df.shape[1])]
    y_tick_labels = [i + 1 for i in range(effect_df.shape[0])]

    # effect_hm = sns.heatmap(effect_df, annot=True, fmt='', ax=ax)
    effect_hm = sns.heatmap(effect_df, annot=effect_df_annot, fmt='', vmin=0, vmax=0.6, ax=ax)
    effect_hm.set(xlabel='Layer', ylabel='Head', xticklabels=x_tick_labels, yticklabels=y_tick_labels,
                  title='Ablation effect per layer and position')
    effect_hm.xaxis.tick_top()
    effect_hm.xaxis.set_label_position('top')

    for idx, (key, value) in enumerate(CONFIG.items()):
        ax.text(1.3, 1 - idx * 0.02, f'{str(key).capitalize()}: {str(value).upper()}',
                transform=ax.transAxes, verticalalignment='top')

    plt.tight_layout()

    plot_filepath = generate_ablation_chart_filepath(CONFIG)

    try:
        plt.savefig(plot_filepath)
        print('Plot saved at', plot_filepath)
    except:
        print('Plot could not be saved')
else:

    effects_df = pd.DataFrame(enumerate(effects), columns=['Layer', 'Effect'])
    effects_df['Layer'] += 1

    before_after_df = pd.DataFrame(data=before_after, columns=['layer', 'original_score', 'ablated_score'])

    before_after_df['layer'] += 1
    before_after_df['original'] = before_after_df['original_score'].apply(lambda x: 'correct' if x > 0 else 'incorrect')
    before_after_df['ablated'] = before_after_df['ablated_score'].apply(lambda x: 'correct' if x > 0 else 'incorrect')
    before_after_df['performance_change'] = before_after_df.apply(lambda row: 'improved' if row['ablated_score'] > row['original_score'] else 'worsened', axis=1)
    before_after_df['category'] = before_after_df.apply(lambda row: f"{row['ablated']}, {row['performance_change']}", axis=1)
    before_after_df
    before_after_summary = before_after_df.\
        groupby(['layer', 'category'])['layer'].\
        count().\
        rename('instances').\
        reset_index().\
        pivot(index='layer', columns='category', values='instances').\
        fillna(0)
    before_after_summary = before_after_summary.div(before_after_summary.sum(axis=1), axis=0)

    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))  # Adjust the figsize as needed

    # First plot
    sns.set_theme(style="whitegrid")
    sns.barplot(data=effects_df, x='Layer', y='Effect', ax=axes[0], color='tomato')

    # Add text annotations
    for idx, (key, value) in enumerate(CONFIG.items()):
        axes[0].text(1.05, 1 - idx * 0.12, f'{str(key).capitalize()}: {str(value).upper()}',
                     transform=axes[0].transAxes, verticalalignment='top')

    axes[0].set_title('Ablation effect per encoder layer', fontsize=16, fontweight='bold')

    # Second plot
    before_after_summary.plot.bar(stacked=True, ax=axes[1], color=['olivedrab', 'steelblue', 'darkred'])
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Proportion of instances')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title='Ablated performance')
    plt.xticks(rotation=0)
    plt.grid(False)
    axes[1].set_title('Distribution of performance after ablation per encoder layer', fontsize=16, fontweight='bold')

    # Tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    # plt.show()

    plot_filepath = generate_ablation_chart_filepath(CONFIG)

    try:
        plt.savefig(plot_filepath)
        print('Plot saved at', plot_filepath)
    except:
        print('Plot could not be saved')
