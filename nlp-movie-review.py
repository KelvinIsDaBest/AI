import streamlit as st
import joblib
from joblib import dump, load # Explicitly import dump and load
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
import zipfile # Import zipfile
import nltk # Import nltk
import matplotlib.pyplot as plt # Import matplotlib for plotting
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Import ConfusionMatrixDisplay


# Download necessary NLTK data
@st.cache_resource # Cache the download
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True) # Corrected resource name
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        st.success("NLTK data downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

download_nltk_data()


# Define the directory where Transformer model is saved (assuming extracted files)
# Or the zip file is located
TRANSFORMER_MODEL_DIR = "./sentiment_model"
TRANSFORMER_ZIP_FILE = "sentiment_model.zip" # Define the name of the zip file if using zip

# --- Function to extract transformer model if needed ---
def extract_transformer_model(zip_path, extract_dir):
    # Check if the expected extracted directory exists and is not empty
    if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
        st.info(f"Transformer model directory not found or is empty. Attempting to extract from {zip_path}...")
        try:
            # Check if the zip file exists at the provided path
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                st.success("Transformer model extracted successfully.")
            else:
                 st.error(f"Error: Transformer zip file not found at {zip_path}.")
        except zipfile.BadZipFile:
            st.error(f"Error: {zip_path} is not a valid zip file.")
        except Exception as e:
            st.error(f"An error occurred during Transformer model extraction: {e}")
    else:
        # st.info("Transformer model directory already exists and is not empty. Skipping extraction.")
        pass # Do nothing if already extracted


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


# --- Load models and data ---

@st.cache_resource # Cache the models to avoid reloading every time
def load_models_and_data():
    models = {}
    data = {}
    try:
        # Call the extraction function for Transformer model first
        extract_transformer_model(TRANSFORMER_ZIP_FILE, TRANSFORMER_MODEL_DIR)

        # Load Transformer model and tokenizer
        if os.path.exists(TRANSFORMER_MODEL_DIR) and os.listdir(TRANSFORMER_MODEL_DIR):
             try:
                 models['transformer_tokenizer'] = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
                 models['transformer_model'] = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
                 models['sentiment_pipeline'] = pipeline("sentiment-analysis", model=models['transformer_model'], tokenizer=models['transformer_tokenizer'])
                 st.write("Transformer model and tokenizer loaded.")
             except Exception as e:
                 st.error(f"Error loading Transformer model and tokenizer: {e}")
                 # Handle case where transformer model loading fails
        else:
             st.warning(f"Transformer model directory not found or is empty at {TRANSFORMER_MODEL_DIR}. Transformer model will not be available.")
             # Handle case where transformer model is not available


        # Load Standard TF-IDF models from the root directory
        models['lr_std_tfidf'] = load('logistic_regression_model_for_std_tfidf_baseline.joblib')
        models['nb_std_tfidf'] = load('naive_bayes_model_for_std_tfidf_baseline.joblib')
        models['svm_std_tfidf'] = load('svm_model_for_std_tfidf_baseline.joblib')
        models['tfidf_vectorizer_std'] = load('tfidf_vectorizer_for_std_tfidf_baseline.joblib')
        st.write("Standard TF-IDF models loaded.")


        # Load POS-Driven models from the root directory
        models['lr_pos_driven'] = load('logistic_regression_model_for_pos_driven.joblib')
        models['nb_pos_driven'] = load('naive_bayes_model_for_pos_driven.joblib')
        models['svm_pos_driven'] = load('svm_model_for_pos_driven.joblib')
        models['tfidf_vectorizer_pos'] = load('tfidf_vectorizer_for_pos_driven.joblib')
        st.write("POS-Driven models loaded.")


        # Load compound_list from the root directory
        try:
            models['compound_list'] = load('compound_list.joblib')
            st.write("Compound list loaded.")
        except FileNotFoundError:
             st.warning(f"Compound list not found at 'compound_list.joblib'. POS-Driven models might not work correctly.")
             models['compound_list'] = [] # Placeholder

        # Load the comparison DataFrame
        data['comparison_df'] = load_comparison_data('comparison_df.pkl')


        # Load true and predicted labels for Confusion Matrices (Assuming they are saved)
        # You need to save these files in your notebook after evaluation
        try:
            data['y_test_std'] = load('lr_predictions_for_std_tfidf_baseline.joblib')
            data['lr_pred_std'] = load('lr_predictions_for_std_tfidf_baseline.joblib')
            data['nb_pred_std'] = load('nb_predictions_for_std_tfidf_baseline.joblib')
            data['svm_pred_std'] = load('svm_predictions_for_std_tfidf_baseline.joblib')

            data['y_test_pos'] = load('y_test_for_pos_driven.joblib')
            data['lr_pred_pos'] = load('lr_predictions_for_pos_driven.joblib')
            data['nb_pred_pos'] = load('nb_predictions_for_pos_driven.joblib')
            data['svm_pred_pos'] = load('svm_predictions_for_pos_driven.joblib')
            transformer_true_labels = load('true_labels.joblib')
            transformer_predicted_labels = load('predicted_labels.joblib')
            label_map_transformer_to_str = {0: 'negative', 1: 'positive'}
            data['true_labels_transformer_str'] = [label_map_transformer_to_str.get(label, 'unknown') for label in transformer_true_labels]
            data['predicted_labels_transformer_str'] = [label_map_transformer_to_str.get(label, 'unknown') for label in transformer_predicted_labels]
            st.write("True and predicted labels loaded for Confusion Matrices.")

        except FileNotFoundError as e:
             st.warning(f"Confusion Matrix data file not found: {e}. Confusion Matrices will not be displayed.")
             # Handle case where confusion matrix data is not available
        except Exception as e:
             st.error(f"Error loading Confusion Matrix data: {e}")


        return models, data
    except FileNotFoundError as e:
        st.error(f"Error loading a model file: {e}. Please ensure all model files are at the repository root or in the specified directories.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models and data: {e}")
        return None, None

# Load the comparison DataFrame
@st.cache_resource # Cache the DataFrame
def load_comparison_data(file_path):
    try:
        comparison_df = pd.read_pickle(file_path)
        st.write("Comparison data loaded.")
        return comparison_df
    except FileNotFoundError:
        st.error(f"Comparison data file not found at {file_path}. The comparison graph will not be displayed.")
        return None
    except Exception as e:
        st.error(f"Error loading comparison data: {e}")
        return None


models, data = load_models_and_data() # Call the combined loading function

if models:
    # --- Streamlit App ---
    st.title("Large-Scale Movie Reviews Sentiment Analysis Through TF-IDF, POS-Driven Phrase-Level Feature Engineering and Transformer")
    st.set_page_config(layout="wide") # Set layout to wide

# Model Performance Comparison plot 
    st.subheader("Model Performance Comparison")
    if data and 'comparison_df' in data and data['comparison_df'] is not None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        plot_df_individual = data['comparison_df'].set_index('Model')[metrics]

        fig, ax = plt.subplots(figsize=(16, 8)) # Adjusted figure size for Streamlit
        bar_width = 0.20
        x = np.arange(len(plot_df_individual.index))

        for i, metric in enumerate(metrics):
            metric_values = plot_df_individual[metric].values
            bars = ax.bar(x + i * bar_width, metric_values, bar_width, label=metric)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if pd.notna(height):
                    ax.annotate(f'{height:.4f}', # Reduced precision for display
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8) # Reduced font size


        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(plot_df_individual.index, rotation=45, ha='right') # , fontsize=8 Adjusted font size
        ax.legend() # loc='lower right' Adjusted legend location

        # Adjust y-axis limits
        min_val = plot_df_individual.min().min() * 0.98 if not plot_df_individual.empty else 0.0
        max_val = plot_df_individual.max().max() * 1.02 if not plot_df_individual.empty else 1.0
        ax.set_ylim(min_val, max_val)

        plt.tight_layout()
        st.pyplot(fig) # Display the plot in Streamlit
    else:
        st.warning("Comparison data not available to display the graph.")

# Confusion Matrices
    st.write("---") # Separator before prediction section
    st.subheader("Confusion Matrices")


    classes = ['negative', 'positive'] # Define classes for confusion matrix

    if data and 'y_test_std' in data and 'lr_pred_std' in data and 'nb_pred_std' in data and 'svm_pred_std' in data and \
            'y_test_pos' in data and 'lr_pred_pos' in data and 'nb_pred_pos' in data and 'svm_pred_pos' in data and \
            'true_labels_transformer_str' in data and 'predicted_labels_transformer_str' in data:

        st.write("### Standard TF-IDF")
        # Create columns within the Standard TF-IDF section for horizontal layout
        col_std_lr, col_std_nb, col_std_svm = st.columns(3)
        with col_std_lr:
            # Clear previous plot
            plt.close('all')
            cm_lr_std = confusion_matrix(data['y_test_std'], data['lr_pred_std'], labels=classes)
            disp_lr_std = ConfusionMatrixDisplay(confusion_matrix=cm_lr_std, display_labels=classes)
            fig_lr_std, ax_lr_std = plt.subplots()
            disp_lr_std.plot(cmap=plt.cm.Blues, ax=ax_lr_std)
            ax_lr_std.set_title('LR (Standard TF-IDF)')
            st.pyplot(fig_lr_std)
        with col_std_nb:
            # Clear previous plot
            plt.close('all')
            cm_nb_std = confusion_matrix(data['y_test_std'], data['nb_pred_std'], labels=classes)
            disp_nb_std = ConfusionMatrixDisplay(confusion_matrix=cm_nb_std, display_labels=classes)
            fig_nb_std, ax_nb_std = plt.subplots()
            disp_nb_std.plot(cmap=plt.cm.Blues, ax=ax_nb_std)
            ax_nb_std.set_title('Naive Bayes (Standard TF-IDF)')
            st.pyplot(fig_nb_std)
        with col_std_svm:
            # Clear previous plot
            plt.close('all')
            cm_svm_std = confusion_matrix(data['y_test_std'], data['svm_pred_std'], labels=classes)
            disp_svm_std = ConfusionMatrixDisplay(confusion_matrix=cm_svm_std, display_labels=classes)
            fig_svm_std, ax_svm_std = plt.subplots()
            disp_svm_std.plot(cmap=plt.cm.Blues, ax=ax_svm_std)
            ax_svm_std.set_title('SVM (Standard TF-IDF)')
            st.pyplot(fig_svm_std)

        st.write("### POS-Driven")
        # Create columns within the POS-Driven section for horizontal layout
        col_pos_lr, col_pos_nb, col_pos_svm = st.columns(3)
        with col_pos_lr:
            # Clear previous plot
            plt.close('all')
            cm_lr_pos = confusion_matrix(data['y_test_pos'], data['lr_pred_pos'], labels=classes)
            disp_lr_pos = ConfusionMatrixDisplay(confusion_matrix=cm_lr_pos, display_labels=classes)
            fig_lr_pos, ax_lr_pos = plt.subplots()
            disp_lr_pos.plot(cmap=plt.cm.Blues, ax=ax_lr_pos)
            ax_lr_pos.set_title('LR (POS-Driven)')
            st.pyplot(fig_lr_pos)
        with col_pos_nb:
            # Clear previous plot
            plt.close('all')
            cm_nb_pos = confusion_matrix(data['y_test_pos'], data['nb_pred_pos'], labels=classes)
            disp_nb_pos = ConfusionMatrixDisplay(confusion_matrix=cm_nb_pos, display_labels=classes)
            fig_nb_pos, ax_nb_pos = plt.subplots()
            disp_nb_pos.plot(cmap=plt.cm.Blues, ax=ax_nb_pos)
            ax_nb_pos.set_title('Naive Bayes (POS-Driven)')
            st.pyplot(fig_nb_pos)
        with col_pos_svm:
            # Clear previous plot
            plt.close('all')
            cm_svm_pos = confusion_matrix(data['y_test_pos'], data['svm_pred_pos'], labels=classes)
            disp_svm_pos = ConfusionMatrixDisplay(confusion_matrix=cm_svm_pos, display_labels=classes)
            fig_svm_pos, ax_svm_pos = plt.subplots()
            disp_svm_pos.plot(cmap=plt.cm.Blues, ax=ax_svm_pos)
            ax_svm_pos.set_title('SVM (POS-Driven)')
            st.pyplot(fig_svm_pos)

        st.write("### Transformer")
        # Create a single column for the Transformer confusion matrix
        col_transformer_plot, col_spacer1, col_spacer2 = st.columns(3) # Transformer in its own row
        with col_transformer_plot:
            # Clear previous plot
            plt.close('all')
            # Transformer
            # Note: Transformer predictions are 0/1, so we need to map them back to string labels or use 0/1 labels for confusion matrix
            # Using string labels for consistency with others
            cm_transformer = confusion_matrix(data['true_labels_transformer_str'], data['predicted_labels_transformer_str'], labels=classes)
            disp_transformer = ConfusionMatrixDisplay(confusion_matrix=cm_transformer, display_labels=classes)
            fig_transformer, ax_transformer = plt.subplots()
            disp_transformer.plot(cmap=plt.cm.Blues, ax=ax_transformer)
            ax_transformer.set_title('Transformer')
            st.pyplot(fig_transformer)


    else:
        st.warning("True and predicted labels not available to display Confusion Matrices. Please ensure the data files are saved and accessible.")


    st.write("---") # Separator before prediction section
    st.subheader("Predict Movie Review")

    user_input = st.text_area("Enter your movie review here:", height=200)

    if st.button("Analyze Sentiment"):
        if user_input:
            st.subheader("Analysis Results:")

            # Perform sentiment analysis with each model
            results = {}

            # Standard TF-IDF models
            # Check if the necessary standard models are loaded
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
                     # Add error results for standard models
                     results['Standard TF-IDF + Logistic Regression'] = {'prediction': 'Error', 'confidence': 0.0}
                     results['Standard TF-IDF + Naive Bayes'] = {'prediction': 'Error', 'confidence': 0.0}
                     results['Standard TF-IDF + SVM'] = {'prediction': 'Error', 'confidence': 0.0}

            else:
                 st.warning("Standard TF-IDF models were not loaded successfully. Skipping predictions for these models.")
                 results['Standard TF-IDF + Logistic Regression'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
                 results['Standard TF-IDF + Naive Bayes'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
                 results['Standard TF-IDF + SVM'] = {'prediction': 'Not Loaded', 'confidence': 0.0}


            # POS-Driven models
            # Check if the necessary POS-Driven models are loaded
            if all(model_name in models for model_name in ['lr_pos_driven', 'nb_pos_driven', 'svm_pos_driven', 'tfidf_vectorizer_pos']) and 'compound_list' in models:
                try:
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
                    nb_prob_pos = models['nb_pos_driven'].predict_proba(pos_tfidf_features)[0] # Corrected typo here
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
                     # Add error results for POS-Driven models
                     results['POS-Driven + Logistic Regression'] = {'prediction': 'Error', 'confidence': 0.0}
                     results['POS-Driven + Naive Bayes'] = {'prediction': 'Error', 'confidence': 0.0}
                     results['POS-Driven + SVM'] = {'prediction': 'Error', 'confidence': 0.0}

            else:
                 st.warning("POS-Driven models were not loaded successfully. Skipping predictions for these models.")
                 results['POS-Driven + Logistic Regression'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
                 results['POS-Driven + Naive Bayes'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
                 results['POS-Driven + SVM'] = {'prediction': 'Not Loaded', 'confidence': 0.0}


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
    st.error("Models could not be loaded. Please ensure models are saved and accessible.")
