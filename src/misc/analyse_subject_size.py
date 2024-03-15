import numpy as np

from src.utils.data_utils import load_dataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

# Experiment configuration
CONFIG = {
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'foil',  # foil/caption
    'segment': None  # correct/ambiguous/incorrect
}

OUTPUT_FORMAT = 'eps' # [eps, png]

segment_threshold = 0.25

dataset = load_dataset(**CONFIG)

rows = []
for data in dataset.values():
    row = (data['segment'], data['subject_area'], data['subject_area_binary'], data['score'])
    rows.append(row)

mydf = pd.DataFrame(rows, columns=['Segment', 'Area', 'Area (binary)', 'Score'])
mydf = mydf.dropna()
mydf.Segment = mydf.Segment.apply(lambda x: x.capitalize())

area_corr = mydf[['Area', 'Score']].corr().iloc[0,1]

threshold_accuracies = []
for threshold in np.arange(0.0,0.61, 0.01):
    filtered_df = mydf[mydf['Area'] > threshold]
    acc = (filtered_df['Score'] > 0).mean()
    instances = len(filtered_df)
    threshold_accuracies.append((threshold, acc, instances))

threshold_df = pd.DataFrame(threshold_accuracies, columns=['Threshold', 'Accuracy', 'Sample Size'])
# print(threshold_df)


if OUTPUT_FORMAT == 'png':

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.scatterplot(data=mydf, x='Area', y='Score', hue='Segment', palette=['#cf5944', '#4497cf', '#44cf69'],
                    ax=ax, zorder=2)
    plt.xlabel('Relative size of subject in image', fontsize=13, fontweight='bold')
    plt.ylabel('Score', fontsize=13, fontweight='bold')
    plt.axhline(0, color='grey', linewidth=3, zorder=1)
    plt.text(0.95, 0.1, f"Correlation: {area_corr:.2f}", transform=ax.transAxes, ha='right', bbox=dict(facecolor='white'), fontsize=14, zorder=3)
    plt.show()
    # plt.savefig('../../output/misc/segmentation/score_vs_subject_size.png')

    # mydf.drop('Area (binary)', axis=1).to_csv('size_vs_score.csv', index=False)

if OUTPUT_FORMAT == 'eps':
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 19
    })

    fig, ax1 = plt.subplots(1, 1)
    sns.scatterplot(data=mydf, x='Area', y='Score', hue='Segment', palette=['#de6c6c', '#f7c094', '#91C788'],
                    ax=ax1, zorder=2, linewidth=0)
    plt.xlabel('Relative size of subject in image')
    plt.ylabel('Classification score')
    plt.axhline(0, color='grey', linewidth=1, zorder=1)
    plt.text(0.6, 0.05, f"$r={area_corr:.2f}$ ", transform=ax1.transAxes, ha='right',
              fontsize=16, zorder=3)

    lines, labels = ax1.get_legend_handles_labels()
    ax1.get_legend().set_visible(False)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    ax2 = plt.gca().twinx()
    sns.lineplot(data=threshold_df, x='Threshold', y='Accuracy', ax=ax2, linewidth=2.7, color='#02456b')
    ax2.set_ylim(0.6, 1)
    ax2.legend(lines, labels, title='Segment', framealpha=1.0, loc='lower right', fontsize=13, title_fontsize=16)

    plt.tight_layout()

    # plt.show()
    plt.savefig('../../output/misc/segmentation/score_vs_subject_size.eps', format='eps')