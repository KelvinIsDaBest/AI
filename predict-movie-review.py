import streamlit as st
import joblib
from joblib import dump, load
import os
import re
import string
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import zipfile
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(
    page_title="🎬 Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: white;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .model-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    .positive-result {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 5px solid #10b981;
    }
    .negative-result {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 5px solid #ef4444;
    }
    .summary-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        min-width: 120px;
    }
    .input-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    .analyze-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .badge-positive {
        background: #dcfce7;
        color: #166534;
    }
    .badge-negative {
        background: #fef2f2;
        color: #dc2626;
    }
    .badge-error {
        background: #fef3c7;
        color: #92400e;
    }
    .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

download_nltk_data()

TRANSFORMER_MODEL_DIR = "./sentiment_model"
TRANSFORMER_ZIP_FILE = "sentiment_model.zip"

def extract_transformer_model(zip_path, extract_dir):
    if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
        try:
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            else:
                 st.error(f"Error: Transformer zip file not found at {zip_path}.")
        except Exception as e:
            st.error(f"An error occurred during Transformer model extraction: {e}")
    else:
        pass

# ALL METHODS (keeping your existing functions)
def handle_negations(tokens):
    negation_words = {'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none',
                     'neither', 'nor', 'cannot', 'cant', 'couldnt', 'shouldnt',
                     'wouldnt', 'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'wasnt', 'werent'}

    negated_negatives = {
        'not_bad': 'good',
        'not_terrible': 'decent',
        'not_awful': 'okay',
        'not_horrible': 'acceptable',
        'not_worst': 'better',
        'not_disappointing': 'satisfying',
        'not_boring': 'engaging',
        'not_poor': 'decent',
        'not_weak': 'strong',
        'not_fail': 'succeed'
    }

    result = []
    i = 0

    while i < len(tokens):
        if tokens[i].lower() in negation_words and i + 1 < len(tokens):
            next_word = tokens[i + 1].lower()
            negation_phrase = f"{tokens[i].lower()}_{next_word}"

            if negation_phrase in negated_negatives:
                result.append(negated_negatives[negation_phrase])
                i += 2
            else:
                result.append(f"NOT_{next_word.upper()}")
                i += 2
        else:
            result.append(tokens[i])
            i += 1

    return result

def preprocess_review_standard_improved(review_text):
    text_without_br = re.sub(r'<br\s*/>', ' ', review_text)
    text_expanded_contractions = contractions.fix(text_without_br)
    lowercased_review = text_expanded_contractions.lower()
    tokenized_review = word_tokenize(lowercased_review)
    tokens_with_negations = handle_negations(tokenized_review)

    cleaned_tokens = []
    for word in tokens_with_negations:
        if word.startswith('NOT_') or word in ['good', 'decent', 'okay', 'acceptable', 'better', 'satisfying', 'engaging', 'strong', 'succeed']:
            cleaned_tokens.append(word)
        else:
            word = re.sub(r'\d+', '', word)
            word = re.sub(r'[^\w\s]', '', word)
            word = re.sub(r'^s$', '', word)
            if word and word.strip():
                cleaned_tokens.append(word.strip())

    stop_words = set(stopwords.words('english'))
    sentiment_stopwords = {'very', 'really', 'quite', 'rather', 'too', 'so', 'not', 'no', 'never'}
    filtered_stopwords = stop_words - sentiment_stopwords

    cleaned_review_without_stopword = []
    for word in cleaned_tokens:
        if word.startswith('NOT_') or word.lower() not in filtered_stopwords:
            cleaned_review_without_stopword.append(word)

    return ' '.join(cleaned_review_without_stopword)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def extract_compound_terms(tokens, compounds):
    result = []
    i = 0
    while i < len(tokens):
        found = False
        for compound in compounds:
            compound_words = compound.split('_')
            if i + len(compound_words) <= len(tokens):
                match = True
                for j in range(len(compound_words)):
                    if tokens[i+j] != compound_words[j]:
                        match = False
                        break
                if match:
                    result.append('_'.join(compound_words))
                    i += len(compound_words)
                    found = True
                    break
        if not found:
            result.append(tokens[i])
            i += 1
    return result

def preprocess_review_pos_driven_improved(review_text, compound_list):
    text_without_br = re.sub(r'<br\s*/>', ' ', review_text)
    text_expanded_contractions = contractions.fix(text_without_br)
    lowercased_review = text_expanded_contractions.lower()
    tokenized_review = word_tokenize(lowercased_review)
    tokens_with_negations = handle_negations(tokenized_review)

    cleaned_tokens = []
    for word in tokens_with_negations:
        if word.startswith('NOT_') or word in ['good', 'decent', 'okay', 'acceptable', 'better', 'satisfying', 'engaging', 'strong', 'succeed']:
            cleaned_tokens.append(word)
        else:
            word = re.sub(r'\d+', '', word)
            word = re.sub(r'[^\w\s]', '', word)
            word = re.sub(r'^s$', '', word)
            if word and word.strip():
                cleaned_tokens.append(word.strip())

    pos_tokens = []
    for word in cleaned_tokens:
        if word.startswith('NOT_'):
            pos_tokens.append((word, 'NEG'))
        else:
            pos_tokens.append((word, 'NN'))

    regular_words = [word for word in cleaned_tokens if not word.startswith('NOT_')]
    if regular_words:
        pos_tagged = pos_tag(regular_words)

        pos_tagged_review = []
        reg_idx = 0
        for word in cleaned_tokens:
            if word.startswith('NOT_'):
                pos_tagged_review.append((word, 'NEG'))
            else:
                if reg_idx < len(pos_tagged):
                    pos_tagged_review.append(pos_tagged[reg_idx])
                    reg_idx += 1
                else:
                    pos_tagged_review.append((word, 'NN'))
    else:
        pos_tagged_review = pos_tokens

    lemmatizer = WordNetLemmatizer()
    sentiment_preserve = {'best', 'worst', 'better', 'worse', 'amazing', 'terrible', 'awful', 'excellent',
                         'good', 'decent', 'okay', 'acceptable', 'satisfying', 'engaging', 'strong', 'succeed'}

    cleaned_review_pos = []
    for word, tag in pos_tagged_review:
        if word.startswith('NOT_') or word.lower() in sentiment_preserve or tag.startswith('JJ') or tag.startswith('RB'):
            cleaned_review_pos.append(word.lower())
        elif tag != 'NEG':
            cleaned_review_pos.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
        else:
            cleaned_review_pos.append(word)

    regular_tokens = [token for token in cleaned_review_pos if not token.startswith('NOT_')]
    negation_tokens = [token for token in cleaned_review_pos if token.startswith('NOT_')]

    if regular_tokens:
        compounds_extracted = extract_compound_terms(regular_tokens, compound_list)
        cleaned_review_with_compounds = negation_tokens + compounds_extracted
    else:
        cleaned_review_with_compounds = negation_tokens

    stop_words = set(stopwords.words('english'))
    sentiment_stopwords = {'very', 'really', 'quite', 'rather', 'too', 'so', 'not', 'no', 'never'}
    filtered_stopwords = stop_words - sentiment_stopwords

    cleaned_review_with_compounds_without_stopword = []
    for word in cleaned_review_with_compounds:
        if word.startswith('NOT_') or word.lower() not in filtered_stopwords:
            cleaned_review_with_compounds_without_stopword.append(word)

    return ' '.join(cleaned_review_with_compounds_without_stopword)

@st.cache_resource
def load_models_and_data():
    models = {}
    data = {}
    
    # Progress bar for loading
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text('🔄 Extracting Transformer model...')
        progress_bar.progress(10)
        extract_transformer_model(TRANSFORMER_ZIP_FILE, TRANSFORMER_MODEL_DIR)

        # Load Transformer model and tokenizer
        if os.path.exists(TRANSFORMER_MODEL_DIR) and os.listdir(TRANSFORMER_MODEL_DIR):
             try:
                 status_text.text('🤖 Loading Transformer model...')
                 progress_bar.progress(25)
                 models['transformer_tokenizer'] = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
                 models['transformer_model'] = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
                 models['sentiment_pipeline'] = pipeline("sentiment-analysis", model=models['transformer_model'], tokenizer=models['transformer_tokenizer'])
             except Exception as e:
                 st.error(f"Error loading Transformer model and tokenizer: {e}")
        else:
             st.warning(f"Transformer model directory not found or is empty at {TRANSFORMER_MODEL_DIR}.")

        status_text.text('📊 Loading Standard TF-IDF models...')
        progress_bar.progress(50)
        models['lr_std_tfidf'] = load('logistic_regression_model_for_std_tfidf_baseline.joblib')
        models['nb_std_tfidf'] = load('naive_bayes_model_for_std_tfidf_baseline.joblib')
        models['svm_std_tfidf'] = load('svm_model_for_std_tfidf_baseline.joblib')
        models['tfidf_vectorizer_std'] = load('tfidf_vectorizer_for_std_tfidf_baseline.joblib')

        status_text.text('🎯 Loading POS-Driven models...')
        progress_bar.progress(75)
        models['lr_pos_driven'] = load('logistic_regression_model_for_pos_driven.joblib')
        models['nb_pos_driven'] = load('naive_bayes_model_for_pos_driven.joblib')
        models['svm_pos_driven'] = load('svm_model_for_pos_driven.joblib')
        models['tfidf_vectorizer_pos'] = load('tfidf_vectorizer_for_pos_driven.joblib')

        status_text.text('📝 Loading compound list...')
        progress_bar.progress(90)
        try:
            models['compound_list'] = load('compound_list.joblib')
        except FileNotFoundError:
             st.warning(f"Compound list not found. Using empty list.")
             models['compound_list'] = []

        status_text.text('✅ All models loaded successfully!')
        progress_bar.progress(100)
        
        # Clear progress indicators
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return models, data
    except FileNotFoundError as e:
        st.error(f"Error loading a model file: {e}")
        progress_bar.empty()
        status_text.empty()
        return None, None
    except Exception as e:
        st.error(f"Error loading models and data: {e}")
        progress_bar.empty()
        status_text.empty()
        return None, None

def create_results_visualization(results):
    """Create a beautiful visualization of results"""
    # Prepare data for visualization
    model_names = []
    predictions = []
    confidences = []
    colors = []
    
    for model_name, result in results.items():
        if result['prediction'] not in ['Error', 'Not Loaded']:
            model_names.append(model_name.replace(' + ', '\n'))
            predictions.append(result['prediction'].upper())
            confidences.append(result['confidence'])
            colors.append('#10b981' if result['prediction'].lower() == 'positive' else '#ef4444')
    
    if model_names:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=confidences,
            marker_color=colors,
            text=[f"{pred}<br>{conf:.3f}" for pred, conf in zip(predictions, confidences)],
            textposition='auto',
            textfont=dict(color='white', size=12),
            hovertemplate='<b>%{x}</b><br>Prediction: %{text}<br>Confidence: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '🎯 Model Predictions Comparison',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#333'}
            },
            xaxis_title="Models",
            yaxis_title="Confidence Score",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333'),
            height=400,
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        fig.update_xaxis(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)

def render_model_result(model_name, result, model_type=""):
    """Render individual model result with enhanced styling"""
    prediction = result['prediction']
    confidence = result['confidence']
    
    if prediction in ['Error', 'Not Loaded']:
        card_class = "model-card"
        icon = "⚠️"
        badge_class = "badge-error"
    elif prediction.lower() == 'positive':
        card_class = "model-card positive-result"
        icon = "✅"
        badge_class = "badge-positive"
    else:
        card_class = "model-card negative-result"
        icon = "❌"
        badge_class = "badge-negative"
    
    # Format confidence based on model type
    if 'SVM' in model_name:
        conf_text = f"Decision Score: {confidence:.4f}"
    elif model_name == 'Transformer':
        conf_text = f"Score: {confidence:.4f}"
    else:
        conf_text = f"Confidence: {confidence:.4f}"
    
    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                <div>
                    <h4 style="margin: 0; color: #333;">{model_name}</h4>
                    <small style="color: #666;">{model_type}</small>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="status-badge {badge_class}">{prediction.upper()}</span>
                <br><small style="color: #666; margin-top: 0.5rem;">{conf_text}</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">🎬 Movie Sentiment Analyzer</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Advanced ML Models for Movie Review Sentiment Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Model Information")
        
        st.markdown("""
        <div class="sidebar-content">
            <h4>📊 Available Models:</h4>
            <ul>
                <li><strong>Standard TF-IDF</strong> - Traditional approach</li>
                <li><strong>POS-Driven</strong> - Enhanced with linguistics</li>
                <li><strong>Transformer</strong> - State-of-the-art neural model</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-content">
            <h4>🎯 Algorithms:</h4>
            <ul>
                <li>Logistic Regression</li>
                <li>Naive Bayes</li>
                <li>Support Vector Machine</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-content">
            <h4>ℹ️ How it works:</h4>
            <p>Enter a movie review and get sentiment predictions from multiple ML models. The system analyzes text using different approaches and provides confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)

    # Load models
    models, data = load_models_and_data()

    if models is None:
        st.error("❌ Unable to load models. Please check your model files.")
        return

    # Input Section
    st.markdown("""
    <div class="input-section">
        <h3 style="color: #333; margin-bottom: 1rem;">✍️ Enter Your Movie Review</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "", 
            height=200,
            placeholder="Write your movie review here... For example: 'This movie was absolutely amazing! The acting was superb and the plot was engaging throughout.'"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer
        st.markdown("### 💡 Sample Reviews:")
        
        sample_reviews = [
            "This movie was absolutely fantastic! Amazing acting and great storyline.",
            "Terrible movie, boring plot and bad acting throughout.",
            "The film was okay, nothing special but not bad either."
        ]
        
        for i, sample in enumerate(sample_reviews):
            if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                user_input = sample
                st.experimental_rerun()

    # Analysis Button
    analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_clicked:
        if user_input.strip():
            with st.spinner('🔄 Analyzing your review...'):
                results = {}

                # Standard TF-IDF Models
                if all(model_name in models for model_name in ['lr_std_tfidf', 'nb_std_tfidf', 'svm_std_tfidf', 'tfidf_vectorizer_std']):
                    try:
                        processed_std = preprocess_review_standard_improved(user_input)
                        std_tfidf_features = models['tfidf_vectorizer_std'].transform([processed_std])

                        lr_pred_std = models['lr_std_tfidf'].predict(std_tfidf_features)[0]
                        lr_prob_std = models['lr_std_tfidf'].predict_proba(std_tfidf_features)[0]
                        results['Standard TF-IDF + Logistic Regression'] = {
                            'prediction': lr_pred_std,
                            'confidence': max(lr_prob_std)
                        }

                        nb_pred_std = models['nb_std_tfidf'].predict(std_tfidf_features)[0]
                        nb_prob_std = models['nb_std_tfidf'].predict_proba(std_tfidf_features)[0]
                        results['Standard TF-IDF + Naive Bayes'] = {
                            'prediction': nb_pred_std,
                            'confidence': max(nb_prob_std)
                        }

                        svm_pred_std = models['svm_std_tfidf'].predict(std_tfidf_features)[0]
                        svm_decision_std = models['svm_std_tfidf'].decision_function(std_tfidf_features)[0]
                        results['Standard TF-IDF + SVM'] = {
                            'prediction': svm_pred_std,
                            'confidence': abs(svm_decision_std)
                        }
                    except Exception as e:
                         st.error(f"Error during Standard TF-IDF model prediction: {e}")
                         results['Standard TF-IDF + Logistic Regression'] = {'prediction': 'Error', 'confidence': 0.0}
                         results['Standard TF-IDF + Naive Bayes'] = {'prediction': 'Error', 'confidence': 0.0}
                         results['Standard TF-IDF + SVM'] = {'prediction': 'Error', 'confidence': 0.0}

                # POS-Driven Models
                if all(model_name in models for model_name in ['lr_pos_driven', 'nb_pos_driven', 'svm_pos_driven', 'tfidf_vectorizer_pos']) and 'compound_list' in models:
                    try:
                        compound_list = models.get('compound_list', [])
                        processed_pos = preprocess_review_pos_driven_improved(user_input, compound_list)
                        pos_tfidf_features = models['tfidf_vectorizer_pos'].transform([processed_pos])

                        lr_pred_pos = models['lr_pos_driven'].predict(pos_tfidf_features)[0]
                        lr_prob_pos = models['lr_pos_driven'].predict_proba(pos_tfidf_features)[0]
                        results['POS-Driven + Logistic Regression'] = {
                            'prediction': lr_pred_pos,
                            'confidence': max(lr_prob_pos)
                        }

                        nb_pred_pos = models['nb_pos_driven'].predict(pos_tfidf_features)[0]
                        nb_prob_pos = models['nb_pos_driven'].predict_proba(pos_tfidf_features)[0]
                        results['POS-Driven + Naive Bayes'] = {
                            'prediction': nb_pred_pos,
                            'confidence': max(nb_prob_pos)
                        }

                        svm_pred_pos = models['svm_pos_driven'].predict(pos_tfidf_features)[0]
                        svm_decision_pos = models['svm_pos_driven'].decision_function(pos_tfidf_features)[0]
                        results['POS-Driven + SVM'] = {
                            'prediction': svm_pred_pos,
                            'confidence': abs(svm_decision_pos)
                        }
                    except Exception as e:
                         st.error(f"Error during POS-Driven model prediction: {e}")
                         results['POS-Driven + Logistic Regression'] = {'prediction': 'Error', 'confidence': 0.0}
                         results['POS-Driven + Naive Bayes'] = {'prediction': 'Error', 'confidence': 0.0}
                         results['POS-Driven + SVM'] = {'prediction': 'Error', 'confidence': 0.0}

                # Transformer Model
                try:
                    if 'sentiment_pipeline' in models and models['sentiment_pipeline']:
                         transformer_result = models['sentiment_pipeline'](user_input)[0]
                         label_map = {0: "negative", 1: "positive"}
                         label_str = transformer_result['label'].replace("LABEL_", "")
                         try:
                             label_id = int(label_str)
                             sentiment_label = label_map.get(label_id, "unknown")
                             results['Transformer'] = {
                                 'prediction': sentiment_label,
                                 'confidence': transformer_result['score']
                             }
                         except ValueError:
                             st.warning(f"Could not parse transformer label ID: {label_str}")
                             results['Transformer'] = {'prediction': 'Error', 'confidence': 0.0}
                    else:
                         results['Transformer'] = {'prediction': 'Not Loaded', 'confidence': 0.0}

                except Exception as e:
                     st.error(f"Error with Transformer prediction: {e}")
                     results['Transformer'] = {'prediction': 'Error', 'confidence': 0.0}

            # Display Results
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Create visualization
            create_results_visualization(results)
            
            # Individual Model Results
            st.markdown("### 🤖 Individual Model Predictions")
            
            # Group results by model type
            standard_models = {k: v for k, v in results.items() if 'Standard TF-IDF' in k}
            pos_models = {k: v for k, v in results.items() if 'POS-Driven' in k}
            transformer_models = {k: v for k, v in results.items() if 'Transformer' in k}
            
            # Display in organized sections
            if standard_models:
                st.markdown("#### 📊 Standard TF-IDF Models")
                for model_name, result in standard_models.items():
                    render_model_result(model_name, result, "Traditional NLP approach")
            
            if pos_models:
                st.markdown("#### 🎯 POS-Driven Models")
                for model_name, result in pos_models.items():
                    render_model_result(model_name, result, "Linguistics-enhanced approach")
            
            if transformer_models:
                st.markdown("#### 🤖 Transformer Model")
                for model_name, result in transformer_models.items():
                    render_model_result(model_name, result, "State-of-the-art neural approach")

            # Overall Summary
            positive_count = sum(1 for result in results.values() if result['prediction'].lower() == 'positive')
            negative_count = sum(1 for result in results.values() if result['prediction'].lower() == 'negative')
            total_predictions = positive_count + negative_count

            if total_predictions > 0:
                positive_percentage = (positive_count / total_predictions) * 100
                negative_percentage = (negative_count / total_predictions) * 100
                
                # Determine overall sentiment
                if positive_count > negative_count:
                    overall_sentiment = "POSITIVE"
                    overall_color = "#10b981"
                    overall_icon = "😊"
                elif negative_count > positive_count:
                    overall_sentiment = "NEGATIVE"
                    overall_color = "#ef4444"
                    overall_icon = "😞"
                else:
                    overall_sentiment = "MIXED"
                    overall_color = "#f59e0b"
                    overall_icon = "😐"
                
                # Summary card
                st.markdown(f"""
                <div class="summary-card" style="border-left: 5px solid {overall_color};">
                    <h2 style="color: {overall_color}; margin-bottom: 1rem;">
                        {overall_icon} Overall Sentiment: {overall_sentiment}
                    </h2>
                    <div class="metric-container">
                        <div class="metric-box" style="border-left: 3px solid #10b981;">
                            <h3 style="color: #10b981; margin: 0;">{positive_count}</h3>
                            <p style="margin: 0; color: #666;">Positive</p>
                            <small style="color: #999;">{positive_percentage:.1f}%</small>
                        </div>
                        <div class="metric-box" style="border-left: 3px solid #ef4444;">
                            <h3 style="color: #ef4444; margin: 0;">{negative_count}</h3>
                            <p style="margin: 0; color: #666;">Negative</p>
                            <small style="color: #999;">{negative_percentage:.1f}%</small>
                        </div>
                        <div class="metric-box" style="border-left: 3px solid #6366f1;">
                            <h3 style="color: #6366f1; margin: 0;">{total_predictions}</h3>
                            <p style="margin: 0; color: #666;">Total Models</p>
                            <small style="color: #999;">Analyzed</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("⚠️ Please enter a movie review to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666; background: rgba(255, 255, 255, 0.8); border-radius: 10px; margin-top: 2rem;">
        <h4>🎬 Movie Sentiment Analyzer</h4>
        <p>Powered by Advanced Machine Learning Models | TF-IDF • POS-Driven • Transformers</p>
        <small>Built with Streamlit, scikit-learn, and Transformers</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
