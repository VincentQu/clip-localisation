from src.utils.data_utils import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Experiment configuration
CONFIG = {
    'dataset': 'rephrased', # standard/rephrased
    'negation': 'foil',  # foil/caption
    'segment': None  # correct/ambiguous/incorrect
}

segment_threshold = 0.25

dataset = load_dataset(**CONFIG)

rows = []
for data in dataset.values():
    row = (data['segment'], data['subject_area'], data['subject_area_binary'], data['score'])
    rows.append(row)

mydf = pd.DataFrame(rows, columns=['Segment', 'Area', 'Area (binary)', 'Score'])

area_corr = mydf[['Area', 'Score']].corr().iloc[0,1]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.scatterplot(data=mydf, x='Area', y='Score', hue='Segment', palette=['#cf5944', '#4497cf', '#44cf69'],
                ax=ax, zorder=2)
plt.xlabel('Relative size of subject in image', fontsize=13, fontweight='bold')
plt.ylabel('Score', fontsize=13, fontweight='bold')
plt.axhline(0, color='grey', linewidth=3, zorder=1)
plt.text(0.95, 0.1, f"Correlation: {area_corr:.2f}", transform=ax.transAxes, ha='right', bbox=dict(facecolor='white'), fontsize=14, zorder=3)
# plt.show()
plt.savefig('../../output/misc/segmentation/score_vs_subject_size.png')