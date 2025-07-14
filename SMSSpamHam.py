import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("SMSSpamCollection", sep = '\t', header= None, names= ['label', 'message'])
print(df.head())

#encodage : transformation en binaire: spam=1, ham=0
df['label'] = LabelEncoder().fit_transform(df['label'])

#Vectorisation des textes
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

#separation des données d entrainement et de test
