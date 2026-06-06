import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
import joblib
import os
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        color: #333;
    }
    
    .metric-card h4 {
        color: #667eea;
        margin-top: 0;
    }
    
    .metric-card p {
        color: #333;
    }
    
    .metric-card ul {
        color: #333;
    }
    
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'vectorizers' not in st.session_state:
    st.session_state.vectorizers = {}
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Utility functions
def preprocess_text(text):
    """Clean and preprocess text"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-z0-9\s]+', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def load_or_create_models(df):
    """Load or create trained models"""
    
    if not st.session_state.models_trained:
        with st.spinner("Training models... This may take a moment"):
            # Preprocess data
            df['cleaned_text'] = df['Text'].apply(preprocess_text)
            
            # Prepare features and target
            X = df['cleaned_text']
            y = df['Score']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train with BoW
            bow_vec = CountVectorizer(max_features=5000, stop_words='english')
            X_train_bow = bow_vec.fit_transform(X_train)
            X_test_bow = bow_vec.transform(X_test)
            
            model_bow = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            model_bow.fit(X_train_bow, y_train)
            
            bow_accuracy = model_bow.score(X_test_bow, y_test)
            
            # Train with TF-IDF
            tfidf_vec = TfidfVectorizer(max_features=5000, stop_words='english')
            X_train_tfidf = tfidf_vec.fit_transform(X_train)
            X_test_tfidf = tfidf_vec.transform(X_test)
            
            model_tfidf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            model_tfidf.fit(X_train_tfidf, y_train)
            
            tfidf_accuracy = model_tfidf.score(X_test_tfidf, y_test)
            
            # Store in session state
            st.session_state.vectorizers['bow'] = bow_vec
            st.session_state.vectorizers['tfidf'] = tfidf_vec
            st.session_state.models['bow'] = model_bow
            st.session_state.models['tfidf'] = model_tfidf
            st.session_state.models_trained = True
            
            return {
                'bow_accuracy': bow_accuracy,
                'tfidf_accuracy': tfidf_accuracy,
                'X_test': X_test,
                'y_test': y_test
            }
    
    return None

def predict_sentiment(text, vectorizer_type):
    """Predict sentiment for given text"""
    
    if not st.session_state.models_trained:
        st.error("Models not trained yet!")
        return None
    
    # Preprocess input text
    cleaned = preprocess_text(text)
    
    # Get vectorizer and model
    vectorizer = st.session_state.vectorizers[vectorizer_type]
    model = st.session_state.models[vectorizer_type]
    
    # Vectorize
    if vectorizer_type == 'bow':
        X = vectorizer.transform([cleaned])
    else:
        X = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    return {
        'prediction': prediction,
        'probabilities': probabilities,
        'confidence': max(probabilities) * 100
    }

def get_sentiment_label(score):
    """Convert score to sentiment label"""
    if score == 1:
        return "Very Negative", "#FF6B6B"
    elif score == 2:
        return "Negative", "#FFA07A"
    elif score == 3:
        return "Neutral", "#FFD700"
    elif score == 4:
        return "Positive", "#90EE90"
    else:  # score == 5
        return "Very Positive", "#32CD32"

# Main App
def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; color: white; margin-bottom: 20px;">
            <h1>Sentiment Analysis Tool</h1>
            <p style="font-size: 16px; margin-top: 10px;">
                Analyze sentiment using Bag of Words or TF-IDF vectorization
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
            <h3>Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        app_mode = st.radio(
            "Select Mode:",
            ["Home", "Train Models", "Predict Sentiment", "Model Performance"],
            key="app_mode"
        )
    
    # Main content
    if app_mode == "Home":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3>How to Use</h3>
                <ol>
                    <li>Go to "Train Models" to train sentiment models</li>
                    <li>Choose between BoW or TF-IDF vectorizers</li>
                    <li>Use "Predict Sentiment" to analyze new reviews</li>
                    <li>Check "Model Performance" for accuracy metrics</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h3>About This Tool</h3>
                <p>This sentiment analysis application uses machine learning to classify 
                text reviews into 5 sentiment categories:</p>
                <ul>
                    <li><strong>1 Star:</strong> Very Negative</li>
                    <li><strong>2 Stars:</strong> Negative</li>
                    <li><strong>3 Stars:</strong> Neutral</li>
                    <li><strong>4 Stars:</strong> Positive</li>
                    <li><strong>5 Stars:</strong> Very Positive</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="success-box">
            <h4>Key Features</h4>
            <ul>
                <li><strong>Two Vectorizers:</strong> Compare Bag of Words and TF-IDF</li>
                <li><strong>Advanced Preprocessing:</strong> Lemmatization and stopword removal</li>
                <li><strong>Real-time Predictions:</strong> Get instant sentiment analysis</li>
                <li><strong>Performance Metrics:</strong> View model accuracy and confidence</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif app_mode == "Train Models":
        st.markdown("## Train Sentiment Models")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <p>Upload a CSV file with columns: <strong>Text</strong> and <strong>Score</strong> (1-5)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Upload Dataset")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate dataset
                required_columns = ['Text', 'Score']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"Dataset must contain columns: {required_columns}")
                else:
                    st.success(f"Dataset loaded successfully! Shape: {df.shape}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Reviews", len(df))
                    with col2:
                        st.metric("Average Score", f"{df['Score'].mean():.2f}")
                    with col3:
                        st.metric("Score Range", f"{df['Score'].min()}-{df['Score'].max()}")
                    
                    st.dataframe(df.head(5))
                    
                    if st.button("Train Models", key="train_btn", use_container_width=True):
                        results = load_or_create_models(df)
                        
                        if results:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="success-box" style="padding: 20px; border-radius: 10px;">
                                    <h4>Bag of Words (BoW)</h4>
                                    <h2 style="color: white;">{results['bow_accuracy']:.2%}</h2>
                                    <p>Accuracy Score</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="success-box" style="padding: 20px; border-radius: 10px; 
                                            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                                    <h4>TF-IDF Vectorizer</h4>
                                    <h2 style="color: white;">{results['tfidf_accuracy']:.2%}</h2>
                                    <p>Accuracy Score</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.success("Models trained successfully!")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Example of expected format
        with st.expander("See example CSV format"):
            example_df = pd.DataFrame({
                'Text': [
                    'This product is amazing, highly recommended!',
                    'Very disappointed with the quality.',
                    'It is okay, nothing special.'
                ],
                'Score': [5, 1, 3]
            })
            st.dataframe(example_df)
    
    elif app_mode == "Predict Sentiment":
        st.markdown("## Predict Sentiment")
        
        if not st.session_state.models_trained:
            st.warning("Models not trained yet! Please go to 'Train Models' first.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("### Vectorizer Selection")
                vectorizer_choice = st.radio(
                    "Choose Vectorizer:",
                    ["Bag of Words (BoW)", "TF-IDF"],
                    key="vec_choice"
                )
                vectorizer_type = 'bow' if vectorizer_choice == "Bag of Words (BoW)" else 'tfidf'
            
            with col1:
                st.markdown("### Enter Your Review")
                review_text = st.text_area(
                    "Type or paste your review here:",
                    placeholder="Enter text to analyze sentiment...",
                    height=150,
                    label_visibility="collapsed"
                )
            
            if review_text:
                if st.button("Analyze Sentiment", use_container_width=True, key="predict_btn"):
                    result = predict_sentiment(review_text, vectorizer_type)
                    
                    if result:
                        st.markdown("---")
                        st.markdown("### Sentiment Analysis Result")
                        
                        sentiment_label, color = get_sentiment_label(result['prediction'])
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col2:
                            st.markdown(f"""
                            <div style="background: {color}; padding: 20px; border-radius: 10px; 
                                        text-align: center; color: white;">
                                <h1 style="color: white; margin: 0;">{result['prediction']} ⭐</h1>
                                <h3 style="color: white; margin-top: 10px;">{sentiment_label}</h3>
                                <p style="color: white; margin-top: 10px;">Confidence: {result['confidence']:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("### Sentiment Distribution")
                        
                        sentiment_labels = ['Very Negative (1)', 'Negative (2)', 'Neutral (3)', 
                                           'Positive (4)', 'Very Positive (5)']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            chart_data = pd.DataFrame({
                                'Sentiment': sentiment_labels,
                                'Probability': result['probabilities'] * 100
                            })
                            st.bar_chart(data=chart_data.set_index('Sentiment'), height=300)
                        
                        with col2:
                            for i, (label, prob) in enumerate(zip(sentiment_labels, result['probabilities'])):
                                st.write(f"{label}: {prob*100:.1f}%")
                        
                        st.markdown("---")
                        st.markdown("### Processed Text")
                        st.write(preprocess_text(review_text))
    
    elif app_mode == "Model Performance":
        if not st.session_state.models_trained:
            st.warning("Models not trained yet! Please go to 'Train Models' first.")
        else:
            st.markdown("## Model Performance Metrics")
            
            st.markdown("""
            <div class="info-box">
                <p>This section shows the performance of both sentiment analysis models.</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["Model Comparison", "Vectorizer Information"])
            
            with tab1:
                st.markdown("### How the Models Work")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 10px 0; color: #333;">
                        <h4 style="color: #667eea; margin-top: 0;">Bag of Words (BoW)</h4>
                        <p style="color: #333;"><strong>How it works:</strong> Counts the frequency of each word in the text.</p>
                        <p style="color: #333;"><strong>Pros:</strong></p>
                        <ul style="color: #333;">
                            <li>Simple and interpretable</li>
                            <li>Fast computation</li>
                            <li>Captures word frequency patterns</li>
                        </ul>
                        <p style="color: #333;"><strong>Cons:</strong></p>
                        <ul style="color: #333;">
                            <li>Ignores word order</li>
                            <li>Common words get high weights</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 10px 0; color: #333;">
                        <h4 style="color: #764ba2; margin-top: 0;">TF-IDF Vectorizer</h4>
                        <p style="color: #333;"><strong>How it works:</strong> Weights words by their importance in the document 
                        relative to the entire corpus.</p>
                        <p style="color: #333;"><strong>Pros:</strong></p>
                        <ul style="color: #333;">
                            <li>Better word weighting</li>
                            <li>Reduces impact of common words</li>
                            <li>Usually better accuracy</li>
                        </ul>
                        <p style="color: #333;"><strong>Cons:</strong></p>
                        <ul style="color: #333;">
                            <li>More computationally intensive</li>
                            <li>Less interpretable</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("### Preprocessing Steps")
                
                st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); color: #333;">
                    <h4 style="color: #667eea; margin-top: 0;">Text Processing Pipeline</h4>
                    <p style="color: #333;"><strong>The text goes through these steps:</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                steps = [
                    "1. Convert text to lowercase",
                    "2. Remove special characters and digits",
                    "3. Tokenization (split into words)",
                    "4. Lemmatization (convert to base form)",
                    "5. Remove stopwords (common words like 'the', 'and')",
                    "6. Vectorization (convert to numbers for ML)"
                ]
                
                cols = st.columns(2)
                for i, step in enumerate(steps):
                    col = cols[i % 2]
                    with col:
                        st.markdown(f"""
                        <div style="background: #f0f0f0; padding: 15px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #667eea;">
                            <p style="color: #333; margin: 0; font-weight: 500;">{step}</p>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
