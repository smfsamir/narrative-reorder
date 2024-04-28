import seaborn as sns
import polars as pl
import matplotlib.pyplot as plt
import json

# Apply the default seaborn theme
sns.set_theme()

# Load shuffled sentence dataset
dataset = [json.loads(line) for line in open("shuffled_sent_data.jsonl")]
labels = [d['label'] for d in dataset]
lengths = [len(d['list_of_sentences']) for d in dataset]

print(len(d['list_of_sentence']) for d in dataset)

frame = pl.DataFrame({
    "label": labels,
    "prompt_length": lengths
})

# Create a visualization
plot = sns.histplot(
    data=frame,
    x="prompt_length",
    stat="count"
).set(title='Prompt Length', xlabel='Number of Sentences')

plt.savefig('histogram.png', dpi=300)