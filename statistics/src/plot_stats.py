import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


file_path = '../data/papers.csv'
data = pd.read_csv(file_path)

colors = ['#c1d08a', '#7cb46b', '#769a6e', '#96845a', '#073B3A']
sns.set(rc={'axes.facecolor': '#b0c1b3', 'figure.facecolor': '#b0c1b3'})

plt.figure(figsize=(12, 10))


# Phrase count
titles = data['Title'].tolist()
phrases = ['training-free', 'zero-cost', 'zero-shot', 'without training']
word_counts = {phrase: sum(1 for title in titles if re.search(r'\b' + re.escape(phrase) + r'\b', title, flags=re.IGNORECASE)) for phrase in phrases}

plt.subplot(221)
plt.bar(word_counts.keys(), word_counts.values(), color=colors[:-1], edgecolor='black')
plt.xlabel('Phrases', fontsize=12)
plt.ylabel('Title Count', fontsize=12)
plt.title('Occurrences of Training-Free NAS Phrases in Research Titles', fontsize=14, y=1.04, fontweight='bold')
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.tight_layout()

# Type
type_counts = data['Type'].str.split(',').explode().str.strip().value_counts()

plt.subplot(222)

percentages = 100 * type_counts / sum(type_counts)
labels = [f"{type_}: {count} ({percentage:.1f}%)" for type_, count, percentage in zip(type_counts.index, type_counts, percentages)]

plt.pie(type_counts, labels=labels, startangle=140, colors=colors, autopct='', pctdistance=0.85)
plt.title('Distribution of Training-Free NAS Algorithm Types', fontsize=14, y=1.04, fontweight='bold')
plt.axis('equal')

plt.tight_layout()


# Word cloud
titles = data['Title']
text = ' '.join(titles)

wordcloud = WordCloud(width=800, height=400, background_color='#b0c1b3').generate(text)
plt.subplot(212) 
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') 
plt.title('Word Cloud of Training-Free NAS', fontsize=14, y=1.04, fontweight='bold')

plt.tight_layout()

plt.savefig("../fig/stats.png", format="png", dpi=300, bbox_inches="tight")
