from src.utils.model_utils import load_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
from tqdm import tqdm

# Load model
model, processor = load_model()
layers = model.text_model.config.num_hidden_layers
attn_heads = model.text_model.config.num_attention_heads
text_model = model.text_model

# Extract vocabulary
stoi = processor.tokenizer.vocab
itos = {i: s for s, i in stoi.items()}

# Load tokeniser
nlp = spacy.load("en_core_web_sm")

# Load dataset
dataset = pd.read_csv('../../data/datasets/cannot/cannot_dataset_v1.1.tsv', sep='\t')

negators = ["no"]
pattern = '|'.join(f'\\b{neg}\\b' for neg in negators)

sentences = dataset[dataset.label==1]
sentences = sentences[sentences.hypothesis.str.contains(pattern, case=False)].hypothesis
sentences = sentences[~sentences.str.contains(r'no[!.,;?-]', case=False, regex=True)]

texts = []
results_tmp = []
for text in tqdm(sentences):
    # text = text + ' not'
    doc = nlp(text)

    for chunk in doc.noun_chunks:
        if chunk[0].text.lower() == "no":  # Check if the first token in the noun chunk is "no"
            # print(f"Case 1: {sentence.replace(' no ', ' some ')}")
            break
    else:
        continue

    text_input = processor.tokenizer(text=text, return_tensors="pt", padding=True)
    out = text_model(**text_input, output_attentions=True)

    text_some = text.lower().replace('no ', 'some ')
    text_input_some = processor.tokenizer(text=text_some, return_tensors="pt", padding=True)
    out_some = text_model(**text_input_some, output_attentions=True)

    if text_input.input_ids.size() == text_input_some.input_ids.size():

        # tokens = [itos[i.item()] for i in text_input.input_ids.squeeze()]
        negator = 'no'
        negator_token = stoi[negator + '</w>']
        negator_token_pos = (text_input.input_ids.squeeze() == negator_token).nonzero().flatten()[0].item()

        att_no = torch.stack([out.attentions[i].squeeze() for i in range(layers)])
        att_some = torch.stack([out_some.attentions[i].squeeze() for i in range(layers)])

        max_attn_diff = np.zeros((attn_heads, layers))
        for l in range(layers):
            for h in range(attn_heads):
                attention_to_no = att_no[l, h, negator_token_pos + 1:, negator_token_pos]
                attention_to_some = att_some[l, h, negator_token_pos + 1:, negator_token_pos]
                attention_diff = attention_to_no - attention_to_some
                max_attn_diff[h, l] = attention_diff.max().item()

        results_tmp.append(max_attn_diff)
        texts.append(text)

result = np.stack(results_tmp).mean(0)
avg_per_layer = result.mean(axis=0)

result_annot = np.array([[f'{x:.2f}'.lstrip('0').replace('-0', '-') for x in row] for row in result])


# Plotting

plt.rcParams.update({
        "text.usetex": True,
        "font.size": 15
    })

x_tick_labels = [i + 1 for i in range(layers)]
y_tick_labels = [i + 1 for i in range(attn_heads)]

fig, axes = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1],
                                                'width_ratios': [20, 1]})

sns.set_theme(style="whitegrid")
attn_hm = sns.heatmap(result, annot=result_annot, fmt='', vmin=0, vmax=0.6, ax=axes[0,0], cbar_ax=axes[0,1])
attn_hm.set(ylabel='Attention head', yticklabels=y_tick_labels)

attn_hm.set_xticks([])

bp = sns.barplot(avg_per_layer, ax=axes[1, 0], color='#02456b', linewidth=0)
bp.set(xlabel='Layer', ylabel='Average', xticklabels=x_tick_labels)
axes[1, 0].set_ylim(0, 0.1)

axes[1, 1].axis('off')

plt.tight_layout(pad=0.5)

# plt.show()

plot_filename_eps = f"attention_cannot.eps"
plot_filepath = f"../../output/charts/attention/{plot_filename_eps}"

plt.savefig(plot_filepath, format='eps')
print(f'Output saved at {plot_filepath}')