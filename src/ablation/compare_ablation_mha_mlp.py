from src.utils.ablation_utils import generate_ablation_result_filepath, generate_ablation_chart_filepath

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

NEGATION = 'caption'
SEGMENT = 'incorrect'

# MHA
MHA_CONFIG = {
    'encoder': 'vision',  # vision/text
    'component': 'mha',  # mha/mlp
    'dataset': 'rephrased',  # standard/rephrased
    'negation': NEGATION,  # foil/caption
    'metric': 'difference',  # absolute/difference
    'segment': SEGMENT,  # correct/ambiguous/incorrect
    'effect': 'absolute'  # absolute/relative
}

mha_h5_filepath = generate_ablation_result_filepath(MHA_CONFIG)

with h5py.File(mha_h5_filepath, 'r') as hdf:
    # Load effects
    mha_effects = np.array(hdf['effects'])

mha_effects_df = pd.DataFrame(enumerate(mha_effects), columns=['Layer', 'MHA'])
mha_effects_df['Layer'] += 1

# MLP
MLP_CONFIG = {
    'encoder': 'vision',  # vision/text
    'component': 'mlp',  # mha/mlp
    'dataset': 'rephrased',  # standard/rephrased
    'negation': NEGATION,  # foil/caption
    'metric': 'difference',  # absolute/difference
    'segment': SEGMENT,  # correct/ambiguous/incorrect
    'effect': 'absolute'  # absolute/relative
}

mlp_h5_filepath = generate_ablation_result_filepath(MLP_CONFIG)

with h5py.File(mlp_h5_filepath, 'r') as hdf:
    # Load effects
    mlp_effects = np.array(hdf['effects'])

mlp_effects_df = pd.DataFrame(enumerate(mlp_effects), columns=['Layer', 'MLP'])
mlp_effects_df['Layer'] += 1

# Merge into one dataframe
comp_df_ = pd.merge(mha_effects_df, mlp_effects_df)
comp_df_ = comp_df_.astype(dtype={'Layer':'object'})
comp_df_.loc['mean'] = comp_df_.mean()
comp_df_.loc['mean', 'Layer'] = 'Mean'

comp_df = pd.melt(comp_df_, id_vars='Layer', value_vars=['MHA', 'MLP'], var_name='Component', value_name='Effect')
comp_df

# Plot
fig, ax = plt.subplots(1, 1, figsize=(6,5))
comp_plot = sns.barplot(comp_df, x = 'Layer', y='Effect', hue='Component', palette=['#345678', '#ffbb55'])
plot_title = f'Ablation effect per layer - Segment: {SEGMENT}, Negation: {NEGATION}'
comp_plot.set(title=plot_title)
plt.tight_layout()
# plt.show()

plot_filepath = f'../../output/charts/ablation/ablation_vision_comparison_{NEGATION}_{SEGMENT}.png'

try:
    plt.savefig(plot_filepath)
    print('Plot saved at', plot_filepath)
except:
    print('Plot could not be saved')