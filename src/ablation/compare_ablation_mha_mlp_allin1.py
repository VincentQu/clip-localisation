from src.utils.ablation_utils import generate_ablation_result_filepath, generate_ablation_chart_filepath

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SEGMENT = 'correct'

NEGATIONS = ['foil', 'caption']
COMPONENTS = ['mha', 'mlp']

dfs = []

for component in COMPONENTS:
    for negation in NEGATIONS:

        CONFIG = {
            'encoder': 'vision',  # vision/text
            'component': component,  # mha/mlp
            'dataset': 'rephrased',  # standard/rephrased
            'negation': negation,  # foil/caption
            'metric': 'difference',  # absolute/difference
            'segment': SEGMENT,  # correct/ambiguous/incorrect
            'effect': 'absolute'  # absolute/relative
        }

        h5_filepath = generate_ablation_result_filepath(CONFIG)

        with h5py.File(h5_filepath, 'r') as hdf:
            # Load effects
            effects = np.array(hdf['effects'])

        col_name = f'{component.upper()} - {negation.capitalize()}'
        df = pd.DataFrame(enumerate(effects), columns=['Layer', col_name])
        df['Layer'] += 1
        df = df.set_index('Layer')

        dfs.append(df)

effects_df_ = pd.concat(dfs, axis=1, sort=False, join='inner').reset_index()

effects_df_ = effects_df_.astype(dtype={'Layer':'object'})
effects_df_.loc['mean'] = effects_df_.mean()
effects_df_.loc['mean', 'Layer'] = 'Avg'

effects_df = pd.melt(effects_df_, id_vars='Layer', value_vars=list(effects_df_.columns)[1:], var_name='Component - Negation', value_name='Effect')

# Plot
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 15
})

fig, ax = plt.subplots(1, 1)
comp_plot = sns.barplot(effects_df, x = 'Layer', y='Effect', hue='Component - Negation', ax=ax,
                        palette=['#33b2d6', '#007fa3', '#ffa866', '#cc5800'])
ax.set_ylabel('Ablation effect')
ax.legend(fontsize=13)

plt.tight_layout()
# plt.show()

plot_filepath_eps = f'../../output/charts/ablation/ablation_vision_comparison_all_in_one.eps'

try:
    plt.savefig(plot_filepath_eps, format='eps')
    print('Plot saved at', plot_filepath_eps)
    print(effects_df_)
except:
    print('Plot could not be saved')

