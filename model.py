import regex as re
import pandas as pd
import numpy as np

# Libraries for solving class imbalance
from imblearn.over_sampling import SMOTE

# Libraries for modelling
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imb_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

# Libraries required for models and vectorizers to run
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def turn_to_string(class_name):
    '''
    Function to turn a class name into string
    '''
    return re.findall(r'(.+?)\s*\(', str(class_name))[0]


def run_models(df: pd.DataFrame, model_lst: list, vectorizer_lst: list, name_lst: list = None,
               pre_processor:str = None,
               over_sampling: bool = False, n_splits: int = 5,
               n_repeats: int = 3, random_state: int = 1, results = None):
    '''
    Function to run the list of ML models with corresponding vectorizer. 
    Returns a dataframe of results
    '''

    vm_combinations = [(vec, mod) for vec in vectorizer_lst for mod in model_lst]
    i = 0 # Counter for name_lst

    # Determine the independedent feature name
    if pre_processor == 'nltk':
        pre_processed_col = 'preprocessed_text_nltk'
    elif pre_processor == 'spacy':
        pre_processed_col = 'preprocessed_text_spacy'
    else:
        pre_processed_col = 'preprocessed_text'

    # Create new dictionary to store results if not passed already
    if results is None:
        results = {'Model': [], 'Vectorizer': [], 'Preprocessor': [],
               'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    # Loop to go over each and every combination of model and vectorizer  
    for vectorizer, model in vm_combinations:

        # If oversampling set to true, includes SMOTE in pipeline
        if over_sampling:
            pipe = imb_pipeline([('vec', vectorizer),
                                 ('smote', SMOTE(random_state=random_state)),
                                 ('mod', model)])
        # Pipeline without SMOTE
        else:
            pipe = Pipeline([('vec', vectorizer),
                             ('mod', model)])

        print("-----------------------------------------------------------------------------------------")
        if name_lst is None:
            print(f"{turn_to_string(model)} with {turn_to_string(vectorizer)} and {pre_processor}")
            results['Model'].append(turn_to_string(model))
            results['Vectorizer'].append(turn_to_string(vectorizer))
        else:
            print(f"{name_lst[i][0]} with {name_lst[i][1]} and {pre_processor}")
            results['Model'].append(name_lst[i][0])
            results['Vectorizer'].append(name_lst[i][1])
        
        # Repeated Stratified K-Fold to generate samples for cross validation
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, 
                                     random_state=random_state)
    
        # List of scorers to be calculated
        scorers = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        # Cross validation
        cv_results = cross_validate(pipe, df[pre_processed_col], df['reference_label'], 
                                scoring=scorers, cv=cv, n_jobs=-1, return_estimator=True)

        # Adding results to dictionary
        results['Preprocessor'].append(pre_processor)
        results['Accuracy'].append(np.mean(cv_results['test_accuracy']))
        results['Precision'].append(np.mean(cv_results['test_precision_weighted']))
        results['Recall'].append(np.mean(cv_results['test_recall_weighted']))
        results['F1 Score'].append(np.mean(cv_results['test_f1_weighted']))

        i+=1 # Increase counter value
        print("Model Complete")
        print("-----------------------------------------------------------------------------------------")

    return results
