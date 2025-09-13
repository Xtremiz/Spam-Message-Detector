import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import Extra as e
from collections import Counter
from wordcloud import WordCloud
df = pd.read_csv('spam.csv', encoding='latin1')
df = df[['v1', 'v2']]
df.rename(columns={"v1": "status", "v2": "Text"}, inplace=True)


le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])
count = df['status'].value_counts()

df['num_character'] = df['Text'].apply(len)


df['num_word'] = df['Text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentence'] = df['Text'].apply(lambda x: len(nltk.sent_tokenize(x)))

df['Transform Text'] = df['Text'].apply(e.transform_text)

wc = WordCloud(width=500, height=500, min_font_size=10,
               background_color='white')
df['Transform Text'] = df['Transform Text'].apply(lambda x: str(x))

list_of_text = df[df['status'] == 1]['Transform Text'].to_list()
corpus = []
for i in list_of_text:
    for j in i.split():
        corpus.append(i)
df = pd.DataFrame(Counter(corpus).most_common(30))
sns.barplot(x=0, y=1, data=df)
plt.xticks(rotation='vertical')
plt.show()

