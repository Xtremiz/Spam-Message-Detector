from nltk.corpus import stopwords
import nltk
import string
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y= []
    for i in text:
        y.append(i)
    text = list(y)
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
           y.append(i)
    return y

print(transform_text("hi my name is fozan ahmed !"))