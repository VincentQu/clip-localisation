from src.utils.ablation_utils import generate_ablation_result_filepath, generate_ablation_chart_filepath

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CONFIG = {
    'encoder': 'vision',    # vision/text
    'component': 'mha',     # mha/mlp
    'dataset': 'rephrased', # standard/rephrased
    'correct': True,        # True/False/F
    'negation': 'caption'      # foil/caption
}

filepath = generate_ablation_result_filepath(CONFIG)

try:
    with h5py.File(filepath, 'r') as hdf:
        # Load activations
        grp_activations = hdf['activations']
        activations = {int(layer): np.array(grp_activations[layer]) for layer in grp_activations}

        # Load effects
        grp_effects = hdf['effects']
        effects = {int(layer): np.array(grp_effects[layer]).item() for layer in grp_effects}

except:
    print(f'Results file with provided configuration does not exist.')


effects_df = pd.DataFrame(list(effects.items()), columns=['Layer', 'Effect'])

plt.figure(figsize=(9, 5))
sns.set_theme(style="whitegrid")
ax = sns.lineplot(data=effects_df, x='Layer', y='Effect')

# Add each key-value pair from CONFIG as a text annotation
for idx, (key, value) in enumerate(CONFIG.items()):
    plt.text(1.05, 1 - idx * 0.07, f'{str(key).capitalize()}: {str(value).upper()}', transform=ax.transAxes, verticalalignment='top')

plt.title('Ablation effect per encoder layer', fontsize = 18)

plt.tight_layout()

plot_filepath = generate_ablation_chart_filepath(CONFIG)

try:
    plt.savefig(plot_filepath)
    print('Plot saved at', plot_filepath)
except:
    print('Plot could not be saved')
