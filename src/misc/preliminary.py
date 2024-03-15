from src.utils.data_utils import load_dataset

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from pprint import pprint
import seaborn as sns

pd.set_option('display.max_columns', None)

dataset = load_dataset(segment=None, negation=None)

df = pd.DataFrame([
    (data['negation'], data['positions'], data['subject_area'], data['similarity_caption_foil'], data['score'], data['correct'])
    for data in dataset.values()
], columns = ['Negation', 'Positions', 'Subject size', 'Similarity', 'Score', 'Correct'])

# sim_corr = df[['Similarity', 'Score']].corr().iloc[0,1]
corrs = df.groupby('Negation')[['Similarity', 'Score']].corr().iloc[[1,3], 0].tolist()

# Plotting

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 11
})

# fig, ax = plt.subplots(1, 1)

sns.set_theme(style="white", font_scale=1.33)
splot = sns.relplot(df, x='Similarity', y='Score', col='Negation', hue='Positions', col_order=['caption', 'foil'], palette='crest')

for i, ax in enumerate(splot.axes[0]):
    plt.text(0.95, 0.1, f"$r={corrs[i]:.2f}$ ", transform=ax.transAxes, ha='right',
             fontsize=12, zorder=3)
    for y in [-2, 0, 2, 4, 6]:
        ax.axhline(y, color='#bbbbbb', linewidth=1, zorder=-1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))

splot.set_axis_labels('Similarity between caption and foil', 'Classification score')
splot.set_titles(template='Negation in {col_name}')

# plt.tight_layout()

plt.show()
# plt.savefig('../../output/misc/score_vs_similarity.eps', format='eps')


## Extra analysis of correlation with the "There are no people outliers" removed
df_nopeople = pd.DataFrame([
    (data['negation'], data['positions'], data['subject_area'], data['similarity_caption_foil'], data['score'], data['correct'])
    for data in dataset.values() if not data['caption'] == 'There are no people.'
], columns = ['Negation', 'Positions', 'Subject size', 'Similarity', 'Score', 'Correct'])

corrs_wo_outliers = df_nopeople.drop(['Negation', 'Correct'], axis=1).corr()
corrs_wo_outliers_grouped = df_nopeople.drop('Correct', axis=1).groupby('Negation').corr()
print(corrs_wo_outliers)
print()
print(corrs_wo_outliers_grouped)