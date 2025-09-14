import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import Extra as e
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
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
df['Transform Text'] = df['Transform Text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score

X = cv.fit_transform(df['Transform Text']).toarray()
y = df['status']

X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)

print(confusion_matrix(y_test, gnb_pred))
print(accuracy_score(y_test, gnb_pred))
print(precision_score(y_test, gnb_pred, average='macro'))  # multi-class safe
