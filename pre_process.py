import string
import regex as re
import pandas as pd

# Library for Spacy Implementation
import spacy

# Libraries for NLTK
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet, stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize


def assign_labels(rating):
    if rating >= 4:
        return 1  # 1 for positive
    elif rating == 3:
        return 0  # 0 for neutral
    elif rating <= 2:
        return -1  # -1 for negative


def create_reference_label(df, rating_col: str):
    temp = df.copy(deep=True)
    temp['reference_label'] = temp[rating_col].apply(assign_labels)
    return temp


# 1. Spacy Implementation
def spacy_preprocess_text(df, col_name, new_col_name='preprocessed_text'):
    temp = df.copy(deep=True)

    # Function to check if a word is a stopword or punctuation
    def is_stop_or_punct(word):
        return word.is_stop or word.is_punct

    # Function to clean doc/sentence
    def tokenize(sentence):
        # Tokenization, Lemmatization and Removal of Stopwords and punctuation
        return " ".join([word.lemma_.lower().strip() for word in sentence if not is_stop_or_punct(word)])

    # Initializing spacy pipeline
    nlp = spacy.load('en_core_web_lg', disable=["tok2vec", 'ner', 'parser'])

    # Removal of punctuation
    temp[new_col_name] = temp[col_name].str.replace('[%s]' % re.escape(string.punctuation), '', regex=True)

    # Removal of non-alphanumeric characters (currencies, non alphabetic characters, etc.)
    cleaned_txt_lst = list(temp[new_col_name].str.replace('[^a-zA-Z0-9 ]', ' ', regex=True))

    # Apply pre-process text
    temp[new_col_name] = [tokenize(doc) for doc in nlp.pipe(cleaned_txt_lst, batch_size=2000, n_process=-1)]

    return temp


# 2. NLTK Implementation

def nltk_preprocess_text(df, col_name, new_col_name='preprocessed_text'):
    stop_words = set(stopwords.words("english"))
    temp = df.copy(deep=True)

    # Lemmatizer Model
    wnl = WordNetLemmatizer()

    # Function to get Parts of Speech for each word to pass to Wordnet Lemmatizer
    def get_pos_for_wnl(word):
        pos_tag = nltk.pos_tag([word])[0][1][0].upper()
        pos_tag_dict = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
        return pos_tag_dict.get(pos_tag, wordnet.NOUN)

    # Function to lemmatize words and remove stopwords
    def preprocess_sentence(sentence):
        return " ".join(list(wnl.lemmatize(word, get_pos_for_wnl(word)).lower() for word in word_tokenize(sentence) if
                             word not in stop_words))

    # Removal of non-alpha numeric characters
    temp[new_col_name] = temp[col_name].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

    # Apply the preprocess_sentence function
    temp[new_col_name] = temp[new_col_name].apply(preprocess_sentence)

    return temp
