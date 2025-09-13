import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import Extra as e
from sklearn.model_selection import train_test_split
cv = CountVectorizer()
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

X = cv.fit_transform(df['Transform Text']).toarray()
y= df['status']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.head())

