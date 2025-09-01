!pip install streamlit -q

import streamlit as st
import joblib
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
import numpy as np # Import numpy here

# Define the directory where models are saved
MODEL_SAVE_DIR = "./saved_models"
TRANSFORMER_MODEL_DIR = "./sentiment_model"

# --- Helper functions for preprocessing (copy from your notebook) ---

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


# --- Load models ---

@st.cache_resource # Cache the models to avoid reloading every time
def load_models():
    models = {}
    try:
        # Load Standard TF-IDF models
        models['lr_std_tfidf'] = joblib.load(os.path.join(MODEL_SAVE_DIR, 'lr_std_tfidf_model.pkl'))
        models['nb_std_tfidf'] = joblib.load(os.path.join(MODEL_SAVE_DIR, 'nb_std_tfidf_model.pkl'))
        models['svm_std_tfidf'] = joblib.load(os.path.join(MODEL_SAVE_DIR, 'svm_std_tfidf_model.pkl'))
        models['tfidf_vectorizer_std'] = joblib.load(os.path.join(MODEL_SAVE_DIR, 'tfidf_vectorizer_std.pkl'))

        # Load POS-Driven models
        models['lr_pos_driven'] = joblib.load(os.path.join(MODEL_SAVE_DIR, 'lr_pos_driven_model.pkl'))
        models['nb_pos_driven'] = joblib.load(os.path.join(MODEL_SAVE_DIR, 'nb_pos_driven_model.pkl'))
        models['svm_pos_driven'] = joblib.load(os.path.join(MODEL_SAVE_DIR, 'svm_pos_driven_model.pkl'))
        models['tfidf_vectorizer_pos'] = joblib.load(os.path.join(MODEL_SAVE_DIR, 'tfidf_vectorizer_pos.pkl'))

        # Load Transformer model and tokenizer
        models['transformer_tokenizer'] = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
        models['transformer_model'] = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
        models['sentiment_pipeline'] = pipeline("sentiment-analysis", model=models['transformer_model'], tokenizer=models['transformer_tokenizer'])


        # Need to recreate compound_list from the training data or save it separately
        # For now, we'll use a placeholder or load it if you saved it.
        # If you didn't save it, you'll need to re-generate it from the training data or a subset.
        # Assuming you saved 'compound_list' during training, load it here:
        # models['compound_list'] = joblib.load(os.path.join(MODEL_SAVE_DIR, 'compound_list.pkl'))
        # If not saved, you might need to recompute it on a small dataset or include it directly.
        # For demonstration, let's assume we have a saved compound_list
        # If you didn't save it, replace this with how you would get it.
        # Example placeholder (replace with actual loading or generation)
        # For the POS-Driven preprocess function to work, compound_list is needed.
        # You need to ensure 'compound_list' is available when calling 'preprocess_review_pos_driven_improved'
        # If you did not save the 'compound_list', you might need to regenerate it
        # or include a pre-computed list in your app.
        # Let's assume you saved it as 'compound_list.pkl'
        try:
            models['compound_list'] = joblib.load(os.path.join(MODEL_SAVE_DIR, 'compound_list.pkl'))
            st.write("Compound list loaded.")
        except FileNotFoundError:
             st.warning("Compound list not found. POS-Driven models might not work correctly.")
             # You might need to add logic here to handle this case,
             # e.g., regenerate the list or use a default empty list.
             models['compound_list'] = [] # Placeholder


        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


models = load_models()

if models:
    # --- Streamlit App ---
    st.title("Movie Review Sentiment Analysis")

    user_input = st.text_area("Enter your movie review here:", height=200)

    if st.button("Analyze Sentiment"):
        if user_input:
            st.subheader("Analysis Results:")

            # Perform sentiment analysis with each model
            results = {}

            # Standard TF-IDF models
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

            # POS-Driven models
            # Ensure compound_list is available here
            compound_list = models.get('compound_list', []) # Get compound list, default to empty if not loaded
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

            # Transformer model
            try:
                if 'sentiment_pipeline' in models and models['sentiment_pipeline']:
                     transformer_result = models['sentiment_pipeline'](user_input)[0]
                     label_map = {0: "negative", 1: "positive"} # Use lowercase for consistency
                     # Extract label ID safely
                     label_str = transformer_result['label'].replace("LABEL_", "")
                     try:
                         label_id = int(label_str)
                         sentiment_label = label_map.get(label_id, "unknown") # Use .get for safety
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



            # Display results
            st.write("---")
            st.subheader("Individual Model Predictions:")
            for model_name, result in results.items():
                sentiment = result['prediction'].upper()
                confidence = result['confidence']

                if 'SVM' in model_name:
                    conf_str = f"Decision Score: {confidence:.4f}"
                elif model_name == 'Transformer':
                    conf_str = f"Score: {confidence:.4f}"
                else:
                    conf_str = f"Confidence: {confidence:.4f}"

                if sentiment == 'POSITIVE':
                    st.markdown(f"✓ **{model_name}**: <span style='color:green'>**{sentiment}**</span> ({conf_str})", unsafe_allow_html=True)
                elif sentiment == 'NEGATIVE':
                    st.markdown(f"✗ **{model_name}**: <span style='color:red'>**{sentiment}**</span> ({conf_str})", unsafe_allow_html=True)
                else:
                    st.markdown(f"  **{model_name}**: {sentiment} ({conf_str})", unsafe_allow_html=True)


            # Overall sentiment summary
            positive_count = sum(1 for result in results.values() if result['prediction'].lower() == 'positive')
            negative_count = sum(1 for result in results.values() if result['prediction'].lower() == 'negative')
            total_predictions = positive_count + negative_count # Only count models that successfully predicted

            st.write("---")
            st.subheader("Overall Summary:")
            st.write(f"Positive predictions: {positive_count}/{total_predictions}")
            st.write(f"Negative predictions: {negative_count}/{total_predictions}")

            if positive_count > negative_count:
                overall = "POSITIVE"
                st.markdown(f"Overall sentiment: <span style='color:green'>**{overall}**</span>", unsafe_allow_html=True)
            elif negative_count > positive_count:
                overall = "NEGATIVE"
                st.markdown(f"Overall sentiment: <span style='color:red'>**{overall}**</span>", unsafe_allow_html=True)
            else:
                overall = "MIXED (TIE)"
                st.markdown(f"Overall sentiment: **{overall}**", unsafe_allow_html=True)


        else:
            st.warning("Please enter a movie review to analyze.")

else:
    st.error("Models could not be loaded. Please check the saved models directory.")
