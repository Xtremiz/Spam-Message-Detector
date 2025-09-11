import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('spam.csv', encoding='latin1')
df = df[['v1','v2']]
df.rename(columns={"v1":"status","v2":"Text"},inplace=True)


le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])
print(df['status'].value_counts())
