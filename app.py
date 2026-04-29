import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Ensure src/ is in the path to import preprocess_text
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import preprocess_text

# Set page config for aesthetics
st.set_page_config(
    page_title="Spam Email Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTextInput>div>div>input {
        background-color: #1E2530;
        color: white;
    }
    .stTextArea>div>div>textarea {
        background-color: #1E2530;
        color: white;
    }
    h1 {
        color: #4A90E2;
        font-family: 'Inter', sans-serif;
    }
    .metric-container {
        background-color: #1E2530;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all saved models and the vectorizer."""
    models = {}
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        model_names = ['logistic_regression', 'svm', 'naive_bayes', 'random_forest']
        for name in model_names:
            path = f'models/{name}_model.pkl'
            if os.path.exists(path):
                models[name.replace('_', ' ').title()] = joblib.load(path)
        return vectorizer, models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

vectorizer, models = load_models()

st.title("📧 Advanced Email Spam Classification System")
st.markdown("Classify your emails as **Spam** or **Ham** (Not Spam) using cutting-edge Machine Learning models.")

if not models:
    st.warning("Models not found. Please ensure you have run the training script.")
    st.stop()

# Sidebar for options
st.sidebar.title("Configuration")
st.sidebar.markdown("Choose your preferred Machine Learning Model:")
model_options = list(models.keys()) + ["Ensemble (All Models)"]
selected_model_name = st.sidebar.selectbox("Model Selection", model_options)

if selected_model_name != "Ensemble (All Models)":
    selected_model = models[selected_model_name]
else:
    selected_model = None

# Tabs for Real-time and Batch processing
tab1, tab2, tab3 = st.tabs(["⚡ Real-time Classification", "📂 Batch Classification", "📊 Model Metrics"])

with tab1:
    st.header("Real-time Email Classification")
    st.markdown("Enter the contents of an email below to classify it instantly.")
    
    email_subject = st.text_input("Email Subject (Optional)")
    email_body = st.text_area("Email Body", height=200)
    
    if st.button("Classify Email", type="primary"):
        if not email_body.strip():
            st.warning("Please enter an email body to classify.")
        else:
            with st.spinner("Analyzing email..."):
                full_text = f"{email_subject} {email_body}"
                clean_text = preprocess_text(full_text)
                vec_text = vectorizer.transform([clean_text])
                
                if selected_model_name == "Ensemble (All Models)":
                    st.markdown("### Ensemble Prediction Results")
                    cols = st.columns(4)
                    predictions = []
                    
                    for idx, (name, model) in enumerate(models.items()):
                        pred = model.predict(vec_text)[0]
                        predictions.append(pred)
                        
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(vec_text)[0]
                            confidence = proba[pred] * 100
                        else:
                            confidence = 100.0
                            
                        label = "Spam" if pred == 1 else "Ham"
                        color = "red" if label == "Spam" else "green"
                        
                        with cols[idx]:
                            st.markdown(f"**{name}**")
                            st.markdown(f"<span style='color:{color}; font-weight:bold;'>{label}</span> ({confidence:.1f}%)", unsafe_allow_html=True)
                    
                    spam_votes = sum(predictions)
                    total_models = len(models)
                    overall_pred = 1 if spam_votes > total_models / 2 else 0
                    overall_label = "Spam" if overall_pred == 1 else "Ham"
                    overall_color = "red" if overall_label == "Spam" else "green"
                    
                    st.markdown("---")
                    st.markdown(f"## 🏆 Overall Consensus: <span style='color:{overall_color}'>{overall_label}</span>", unsafe_allow_html=True)
                    st.markdown(f"*{spam_votes} out of {total_models} models voted Spam.*")
                    
                else:
                    prediction = selected_model.predict(vec_text)[0]
                    
                    if hasattr(selected_model, "predict_proba"):
                        proba = selected_model.predict_proba(vec_text)[0]
                        confidence = proba[prediction] * 100
                    else:
                        confidence = 100.0 # Some models like LinearSVC don't have predict_proba by default
                    
                    label = "Spam" if prediction == 1 else "Ham"
                    color = "red" if label == "Spam" else "green"
                    
                    st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                    st.progress(int(confidence))
                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                
with tab2:
    st.header("Batch Email Classification")
    st.markdown("Upload a CSV file containing an `email` or `text` column to classify multiple emails at once.")
    
    uploaded_file = st.file_uploader("Upload CSV or Excel File", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Try to find the text column
        text_col = None
        for col in ['text', 'email', 'message', 'body', 'content']:
            if col in df.columns.str.lower():
                text_col = df.columns[df.columns.str.lower() == col][0]
                break
                
        if text_col is None:
            st.error("Could not find a valid text column. Please ensure your CSV has a column named 'text', 'email', 'message', or 'body'.")
        else:
            if st.button("Run Batch Classification", type="primary"):
                with st.spinner("Classifying emails..."):
                    df['clean_text'] = df[text_col].apply(lambda x: preprocess_text(str(x)))
                    vec_text = vectorizer.transform(df['clean_text'])
                    
                    if selected_model_name == "Ensemble (All Models)":
                        pred_cols = []
                        for name, model in models.items():
                            df[f'{name}_Label'] = model.predict(vec_text)
                            df[f'{name}_Label'] = df[f'{name}_Label'].map({0: 'Ham', 1: 'Spam'})
                            pred_cols.append(f'{name}_Label')
                            
                        df['Spam_Votes'] = (df[pred_cols] == 'Spam').sum(axis=1)
                        df['Ensemble_Label'] = df['Spam_Votes'].apply(lambda x: 'Spam' if x > len(models)/2 else 'Ham')
                        
                        df = df.drop(columns=['clean_text'])
                    else:
                        df['prediction'] = selected_model.predict(vec_text)
                        df['prediction_label'] = df['prediction'].map({0: 'Ham', 1: 'Spam'})
                        
                        # Drop intermediate columns
                        df = df.drop(columns=['clean_text', 'prediction'])
                    
                    st.success("Batch classification complete!")
                    st.dataframe(df.head(10))
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='classified_emails.csv',
                        mime='text/csv',
                    )

with tab3:
    st.header("Model Evaluation Metrics")
    st.markdown("View the performance of the trained models on the test dataset.")
    
    try:
        results_df = pd.read_csv('models/evaluation_results.csv', index_col=0)
        st.dataframe(results_df.style.highlight_max(axis=0, color='#1e5631'))
        
        st.markdown("### Metrics Explained")
        st.markdown("""
        - **Accuracy:** The percentage of correctly predicted emails out of all emails.
        - **Precision:** Out of all emails predicted as Spam, how many were actually Spam. (Important to minimize false positives).
        - **Recall:** Out of all actual Spam emails, how many did we correctly predict.
        - **F1 Score:** The harmonic mean of Precision and Recall. A balanced metric.
        """)
    except FileNotFoundError:
        st.info("Evaluation results not found. They will appear here after training.")
