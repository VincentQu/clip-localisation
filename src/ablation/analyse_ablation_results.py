from src.utils.ablation_utils import generate_ablation_result_filepath, generate_ablation_chart_filepath

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


CONFIG = {
    'encoder': 'vision',    # vision/text
    'component': 'mha',     # mha/mlp
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'caption',  # foil/caption
    'metric': 'difference', # absolute/difference
    'segment': 'correct',  # correct/ambiguous/incorrect
    'effect': 'absolute'    # absolute/relative
}

filepath = generate_ablation_result_filepath(CONFIG)
# print(filepath)

try:
    with h5py.File(filepath, 'r') as hdf:
        # Load effects
        grp_effects = hdf['effects']
        effects = {int(layer): np.array(grp_effects[layer]).item() for layer in grp_effects}

        grp_before_after = hdf['before_after']
        before_after = {int(layer): [(x, y) for x, y in grp_before_after[layer]] for layer in grp_effects}

except:
    print(f'Results file with provided configuration does not exist.')


effects_df = pd.DataFrame(list(effects.items()), columns=['Layer', 'Effect'])
effects_df['Layer'] += 1

before_after_df = pd.DataFrame(
    data=[(layer, orig, ablat) for layer, scores in before_after.items() for orig, ablat in scores],
    columns=['layer', 'original_score', 'ablated_score']
)

before_after_df['layer'] += 1
before_after_df['original'] = before_after_df['original_score'].apply(lambda x: 'correct' if x > 0 else 'incorrect')
before_after_df['ablated'] = before_after_df['ablated_score'].apply(lambda x: 'correct' if x > 0 else 'incorrect')
before_after_df['performance_change'] = before_after_df.apply(lambda row: 'improved' if row['ablated_score'] > row['original_score'] else 'worsened', axis=1)
before_after_df['category'] = before_after_df.apply(lambda row: f"{row['ablated']}, {row['performance_change']}", axis=1)

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
