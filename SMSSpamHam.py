import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import streamlit as st

df = pd.read_csv("SMSSpamCollection", sep = '\t', header= None, names= ['label', 'message'])
print(df.head())

#encodage : transformation en binaire: spam=1, ham=0
df['label'] = LabelEncoder().fit_transform(df['label'])

#Vectorisation des textes
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

#separation des données d entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#entrainement du modele
model = MultinomialNB()
model.fit(X_train, y_train)

#Prediction
y_pred = model.predict(X_test)

#Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#test manuel
sample = ["Congratulations! You'vre won a freee iPhone. CLick here to claim now!",
"Salut, peux-tu me rappeler dès que possible?"]

sample_vectorized = vectorizer.transform(sample)
predictions = model.predict(sample_vectorized)

for text, pred in zip(sample, predictions):
    print(f"Message: {text} \n=> {'Spam' if pred == 1 else 'Ham'}\n")

#sauvegarde du modele
joblib.dump(model, 'spam_classifer.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

#interface graphique
# Remplacer la partie Streamlit par :
st.title("Détecteur de Spam")
user_input = st.text_area("Collez votre email ici", height=200)

if st.button("Analyser"):
    # Vectoriser l'input
    input_vectorized = vectorizer.transform([user_input])
    # Faire la prédiction
    prediction = model.predict(input_vectorized)[0]
    # Afficher le résultat
    st.success(f"Résultat : {'SPAM' if prediction == 1 else 'HAM'}")