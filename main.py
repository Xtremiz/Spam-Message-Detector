import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('spam.csv', encoding='latin1')
print(df.head())
df = df[['v1','v2']]
df.rename({"v1":"status","v2":"Text"})
print(df.info())
