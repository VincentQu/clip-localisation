from src.utils.data_utils import load_dataset

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


CONFIG = {
    'type': 'rephrased',
    'negation': None,
    'segment': None
}
dataset = load_dataset(**CONFIG)

scores = [data['score'] for data in dataset.values()]
segment_data = pd.DataFrame({'score': scores})
segment_data['Segment'] = pd.cut(segment_data['score'], bins=[float('-inf'), -1, 1, float('inf')], labels=['Incorrect', 'Ambiguous', 'Correct'])

bin_size = 0.25
bin_edges = [i * bin_size - bin_size for i in range(int(min(segment_data['score']) / bin_size), int(max(segment_data['score']) / bin_size) + 2)]

segment_data['bin'] = pd.cut(segment_data['score'], bins=bin_edges, labels=bin_edges[:-1], include_lowest=True)

bins = np.arange(start=min(segment_data['bin']), stop=max(segment_data['bin']), step=0.25)

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 17
})

fig, ax = plt.subplots(1, 1)

sns.histplot(data=segment_data, x='score', bins=bins, hue='Segment',
             palette=['#de6c6c', '#f7c094', '#91C788'], ax=ax, alpha=1, edgecolor='white')
ax.set(xlabel='Classification Score', ylabel='Instances')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# plt.show()

plt.tight_layout()

plot_filepath_eps = '../../output/misc/segmentation/segmentation_distribution.eps'

fig.savefig(plot_filepath_eps, format='eps')