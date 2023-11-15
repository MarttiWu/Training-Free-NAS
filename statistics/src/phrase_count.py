import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '../data/papers.csv'
data = pd.read_csv(file_path)

titles = data['Title'].tolist()

phrases = ['training-free', 'zero-cost', 'zero-shot', 'without training']
word_counts = {phrase: sum(1 for title in titles if re.search(r'\b' + re.escape(phrase) + r'\b', title, flags=re.IGNORECASE)) for phrase in phrases}

colors = ['#c1d08a', '#7cb46b', '#769a6e', '#96845a']
sns.set(rc={'axes.facecolor':'#b0c1b3', 'figure.facecolor':'#b0c1b3'})

plt.figure(figsize=(10, 6))
plt.bar(word_counts.keys(), word_counts.values(), color=colors, edgecolor='black')

plt.xlabel('Phrases', fontsize=12)
plt.ylabel('Title Count', fontsize=12)
plt.title('Occurrences of Training-Free NAS Phrases in Research Paper Titles', fontsize=14)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.tight_layout()

plt.savefig("../fig/phrase_count.pdf", format="pdf", bbox_inches="tight")

for phrase, count in word_counts.items():
    print(f'Occurrences of "{phrase}" in titles: {count}')