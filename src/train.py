import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess_text

def train_models():
    # Load data
    try:
        df = pd.read_excel('../data/email_spam_dataset.xlsx')
    except FileNotFoundError:
        print("Dataset not found.")
        return

    print(f"Loaded {len(df)} samples.")
    
    # Preprocess text
    print("Preprocessing text...")
    df['clean_text'] = df['Message'].apply(preprocess_text)
    
    # Map labels to binary
    df['label_num'] = df['Category'].map({'ham': 0, 'spam': 1})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label_num'], test_size=0.2, random_state=42
    )
    
    # TF-IDF Vectorization
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True, kernel='linear'),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    best_model = None
    best_f1 = 0
    best_model_name = ""
    
    os.makedirs('../models', exist_ok=True)
    
    # Save vectorizer
    joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = model.predict(X_test_vec)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
        
        print(f"{name} Results - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Save all models
        joblib.dump(model, f'../models/{name.replace(" ", "_").lower()}_model.pkl')
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with F1-score: {best_f1:.4f}")
    
    # Save summary
    pd.DataFrame(results).T.to_csv('../models/evaluation_results.csv')
    print("Models and evaluation results saved to 'models/' directory.")

if __name__ == "__main__":
    # Ensure working directory is src/
    if not os.path.exists('preprocess.py') and os.path.exists('src/preprocess.py'):
        os.chdir('src')
    train_models()
