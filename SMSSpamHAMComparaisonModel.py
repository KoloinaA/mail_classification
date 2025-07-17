import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement et préparation des données
@st.cache_data
def load_data():
    df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])
    df['label'] = LabelEncoder().fit_transform(df['label'])
    return df

df = load_data()

# 2. Séparation des données
X = df['message']  # Texte brut
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Définition des modèles à comparer
def get_models():
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True, kernel='linear'),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    return models

# 4. Fonction d'évaluation
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    
    for name, model in models.items():
        try:
            # Création du pipeline
            pipeline = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('classifier', model)
            ])
            
            # Entraînement
            pipeline.fit(X_train, y_train)
            
            # Prédiction
            y_pred = pipeline.predict(X_test)
            
            # Évaluation
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score'],
                'model': pipeline
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Erreur avec {name}: {str(e)}")
            continue
    
    return results

# 5. Interface Streamlit
def main():
    st.title("🔄 Comparaison d'Algorithmes de Détection de Spam")
    
    # Chargement des modèles
    models = get_models()
    
    if st.button("Lancer la comparaison des modèles"):
        with st.spinner('Entraînement des modèles en cours...'):
            results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
        
        if results:
            # Affichage des résultats
            st.subheader("Résultats de la comparaison")
            
            # DataFrame des résultats
            result_df = pd.DataFrame.from_dict(results, orient='index')
            st.dataframe(result_df[['accuracy', 'precision', 'recall', 'f1']]
                         .sort_values('accuracy', ascending=False)
                         .style.format("{:.2%}"))
            
            # Visualisation
            fig, ax = plt.subplots(figsize=(10, 6))
            result_df['accuracy'].sort_values().plot(kind='barh', ax=ax, color='teal')
            ax.set_xlabel('Accuracy Score')
            ax.set_xlim(0, 1.0)
            st.pyplot(fig)
            
            # Sauvegarde du meilleur modèle
            best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
            best_model = results[best_model_name]['model']
            joblib.dump(best_model, 'best_model.pkl')
            st.success(f"Meilleur modèle: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.2%})")
    
    # Test manuel
    st.subheader("Testez votre modèle")
    user_input = st.text_area("Entrez un message à analyser:", "")
    
    if st.button("Prédire"):
        try:
            model = joblib.load('best_model.pkl')
            prediction = model.predict([user_input])[0]
            proba = model.predict_proba([user_input])[0]
            
            if prediction == 1:
                st.error(f"SPAM (confiance: {proba[1]:.2%})")
            else:
                st.success(f"HAM (confiance: {proba[0]:.2%})")
        except:
            st.warning("Veuillez d'abord entraîner les modèles")

if __name__ == "__main__":
    main()