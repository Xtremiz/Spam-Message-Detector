import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import Extra as e
df = pd.read_csv('spam.csv', encoding='latin1')
df = df[['v1','v2']]
df.rename(columns={"v1":"status","v2":"Text"},inplace=True)


le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])
count = df['status'].value_counts()

df['num_character'] = df['Text'].apply(len)


df['num_word'] = df['Text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sentence'] = df['Text'].apply(lambda x:len(nltk.sent_tokenize(x)))



