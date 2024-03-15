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
    'component': 'mlp',     # mha/mlp
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'caption',  # foil/caption
    'metric': 'difference', # absolute/difference
    'segment': 'correct',  # correct/ambiguous/incorrect
    'effect': 'absolute'    # absolute/relative
}

OUTPUT_FORMAT = 'eps' # [eps, png]

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

    effect_sum_df = pd.DataFrame(enumerate(effect_df.sum()), columns=['Layer', 'Effect'])
    effect_sum_df['Layer'] += 1
    effect_sum_df

    fig, ax = plt.subplots(2, 2, figsize=(8, 11), gridspec_kw={'height_ratios': [4, 1],
                                                               'width_ratios': [15, 1]})

    x_tick_labels = list(range(1, effect_df.shape[1] + 1))
    y_tick_labels = list(range(1, effect_df.shape[0], 2))
    x_ticks = [i - 1 for i in x_tick_labels]
    y_ticks = [i - 0.5 for i in y_tick_labels]

    effect_hm = sns.heatmap(effect_df, annot=effect_df_annot, fmt='', vmin=0, vmax=0.6, ax=ax[0, 0], cbar_ax=ax[0,1])
    effect_hm.set(xlabel='Layer', ylabel='Position', yticks=y_ticks,
                  xticklabels=x_tick_labels, yticklabels=y_tick_labels,
                  title='Ablation effect per layer and position')
    effect_hm.xaxis.tick_top()
    effect_hm.xaxis.set_label_position('top')

    for idx, (key, value) in enumerate(CONFIG.items()):
        ax[1,1].text(0, 1 - idx * 0.1, f'{str(key).capitalize()}: {str(value).upper()}',
                transform=ax[1,1].transAxes, verticalalignment='top')

    effect_sum_hm = sns.barplot(data=effect_sum_df, x='Layer', y='Effect', color='tomato', ax=ax[1,0])
    ax[1,1].axis('off')
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
    before_after_summary.index = before_after_summary.index.astype(int)

    if OUTPUT_FORMAT == 'png':
        # Create a figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(9, 6))

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
        plt.show()

        plot_filepath = generate_ablation_chart_filepath(CONFIG)

        try:
            plt.savefig(plot_filepath)
            print('Plot saved at', plot_filepath)
        except:
            print('Plot could not be saved')

    if OUTPUT_FORMAT == 'eps':

        plt.rcParams.update({
            "text.usetex": True,
            "font.size": 11
        })

        fig, axes = plt.subplots(2, 1)

        # First plot
        sns.set_theme(style="whitegrid")
        sns.barplot(data=effects_df, x='Layer', y='Effect', ax=axes[0], color='#02456b')

        axes[0].set_title('(a)')
        # axes[0].text(0.5, 0.9, "(a)", ha='center', va='center', transform=axes[0].transAxes, fontweight='bold')

        # Second plot
        before_after_summary.plot.bar(stacked=True, ax=axes[1], color=['#5ea865', '#9ecaa2', '#deede0'], width=0.8)
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Proportion of instances')
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].legend(bbox_to_anchor=(0.92, 1.0), loc='upper right', title='Ablated performance')
        # axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Set x-axis tick labels as integers
        plt.xticks(rotation=0)
        plt.grid(False)
        axes[1].set_title('(b)')
        # axes[1].text(0.5, 0.9, "(b)", ha='center', va='center', transform=axes[1].transAxes)

        # Tight layout for better spacing
        plt.tight_layout(pad=0.5)

        print(effects_df)
        # print(before_after_summary)

        # plt.show()

        plot_filepath_eps = generate_ablation_chart_filepath(CONFIG).replace('.png', '.eps')

        try:
            plt.savefig(plot_filepath_eps, format='eps')
            print('Plot saved at', plot_filepath_eps)
        except:
            print('Plot could not be saved')