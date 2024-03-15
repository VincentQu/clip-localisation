from src.utils.ablation_utils import generate_ablation_result_filepath, generate_ablation_chart_filepath

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

COMPONENT = 'mha'

# Experiment configuration
CONFIG_FOIL = {
    'encoder': 'vision',    # vision/text
    'component': COMPONENT,     # mha/mlp
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'foil',  # foil/caption
    'metric': 'difference', # absolute/difference
    'segment': 'correct',  # correct/ambiguous/incorrect
    'effect': 'absolute'    # absolute/relative
}

h5_filepath_foil = generate_ablation_result_filepath(CONFIG_FOIL)

with h5py.File(h5_filepath_foil, 'r') as hdf:
    # Load effects
    effects_foil = np.array(hdf['effects'])
    # Load before after
    ba_foil = np.array(hdf['before_after'])

ba_foil_df = pd.DataFrame(data=ba_foil, columns=['layer', 'original_score', 'ablated_score'])

ba_foil_df['layer'] += 1
ba_foil_df['original'] = ba_foil_df['original_score'].apply(lambda x: 'correct' if x > 0 else 'incorrect')
ba_foil_df['ablated'] = ba_foil_df['ablated_score'].apply(lambda x: 'correct' if x > 0 else 'incorrect')
ba_foil_df['performance_change'] = ba_foil_df.apply(lambda row: 'improved' if row['ablated_score'] > row['original_score'] else 'worsened', axis=1)
ba_foil_df['category'] = ba_foil_df.apply(lambda row: f"{row['ablated']}, {row['performance_change']}", axis=1)
ba_foil_df
ba_foil_summary = ba_foil_df.\
    groupby(['layer', 'category'])['layer'].\
    count().\
    rename('instances').\
    reset_index().\
    pivot(index='layer', columns='category', values='instances').\
    fillna(0)
ba_foil_summary = ba_foil_summary.div(ba_foil_summary.sum(axis=1), axis=0)
ba_foil_summary.index = ba_foil_summary.index.astype(int)


CONFIG_CAPTION = {
    'encoder': 'vision',    # vision/text
    'component': COMPONENT,     # mha/mlp
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'caption',  # foil/caption
    'metric': 'difference', # absolute/difference
    'segment': 'correct',  # correct/ambiguous/incorrect
    'effect': 'absolute'    # absolute/relative
}

h5_filepath_caption = generate_ablation_result_filepath(CONFIG_CAPTION)

with h5py.File(h5_filepath_caption, 'r') as hdf:
    # Load effects
    effects_caption = np.array(hdf['effects'])
    # Load before after
    ba_caption = np.array(hdf['before_after'])

ba_caption_df = pd.DataFrame(data=ba_caption, columns=['layer', 'original_score', 'ablated_score'])

ba_caption_df['layer'] += 1
ba_caption_df['original'] = ba_caption_df['original_score'].apply(lambda x: 'correct' if x > 0 else 'incorrect')
ba_caption_df['ablated'] = ba_caption_df['ablated_score'].apply(lambda x: 'correct' if x > 0 else 'incorrect')
ba_caption_df['performance_change'] = ba_caption_df.apply(lambda row: 'improved' if row['ablated_score'] > row['original_score'] else 'worsened', axis=1)
ba_caption_df['category'] = ba_caption_df.apply(lambda row: f"{row['ablated']}, {row['performance_change']}", axis=1)
ba_caption_df
ba_caption_summary = ba_caption_df.\
    groupby(['layer', 'category'])['layer'].\
    count().\
    rename('instances').\
    reset_index().\
    pivot(index='layer', columns='category', values='instances').\
    fillna(0)
ba_caption_summary = ba_caption_summary.div(ba_caption_summary.sum(axis=1), axis=0)
ba_caption_summary.index = ba_caption_summary.index.astype(int)


plt.rcParams.update({
    "text.usetex": True,
    "font.size": 9
})

fig, ax = plt.subplots(2, 2, figsize=(6.4, 3.8), gridspec_kw={'height_ratios': [3, 2]})

sns.barplot(data=effects_foil, color='#02456b', linewidth=2.5, ax=ax[0,0])
ax[0,0].set_xticks([])
ax[0,0].set_ylabel(f'{COMPONENT.upper()} Ablation effect')
ax[0,0].set_title('Negation in foil', fontsize=10)
ba_foil_summary.plot.bar(stacked=True, ax=ax[1,0], color=['#72b278', '#9ecaa2', '#deede0'], width=0.8)
ax[1,0].set_xlabel('Layer')
ax[1,0].tick_params(axis='x', labelrotation = 0)
ax[1,0].set_ylabel('Proportion of instances')
ax[1,0].legend().set_visible(False)

sns.barplot(data=effects_caption, color='#02456b', linewidth=2.5, ax=ax[0,1])
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,1].set_title('Negation in caption', fontsize=10)
ba_caption_summary.plot.bar(stacked=True, ax=ax[1,1], color=['#72b278', '#9ecaa2', '#deede0'], width=0.8)
ax[1,1].set_yticks([])
ax[1,1].set_xlabel('Layer')
ax[1,1].tick_params(axis='x', labelrotation = 0)
# ax[1,1].legend(title='', loc='upper right', fontsize=9)
ax[1,1].legend().set_visible(False)

handles, labels = ax[1,1].get_legend_handles_labels()
ax[0,0].legend(reversed(handles), reversed(labels), loc='upper right', fontsize=9)

ymin0, ymax0 = ax[0,0].get_ylim()
ymin1, ymax1 = ax[0,1].get_ylim()
ax[0,0].set_ylim(min(ymin0, ymin1), max(ymax0, ymax1))
ax[0,1].set_ylim(min(ymin0, ymin1), max(ymax0, ymax1))

ax[1,0].yaxis.set_major_formatter(PercentFormatter(1))
ax[1,1].yaxis.set_major_formatter(PercentFormatter(1))

plt.tight_layout(pad=0.8)

plot_filepath_eps = f'../../output/charts/ablation/ablation_{COMPONENT}_2in1.eps'
# plt.show()
plt.savefig(plot_filepath_eps, format='eps')
print(f'Plot saved at {plot_filepath_eps}')
