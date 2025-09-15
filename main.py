import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import Extra as e
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
tfid = TfidfVectorizer(max_features=3000)
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
df['Transform Text'] = df['Transform Text'].apply(lambda x: ' '.join(x))

X = tfid.fit_transform(df['Transform Text']).toarray()
y = df['status']

X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Multinomia nb")
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnb_pred = mnb.predict(X_test)
"""print(confusion_matrix(y_test, mnb_pred))
print(accuracy_score(y_test, mnb_pred))
print(precision_score(y_test, mnb_pred, average='macro')) """

user_input= input("Enter your message for evaluation : ")
vectorization = tfid.transform([user_input]).toarray()
prediction = mnb.predict(vectorization)

if prediction == 0:
    print("your message is ham")
else:
    print("your message is spam")