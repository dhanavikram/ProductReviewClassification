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
    return re.findall(r'(.+?)\s*\(', str(class_name))[0]


def run_models(df: pd.DataFrame, model_lst: list, vectorizer_lst: list,
               over_sampling: bool = False, n_splits: int = 5,
               n_repeats: int = 3, random_state: int = 1):

    vm_combinations = [(vec, mod) for vec in vectorizer_lst for mod in model_lst]

    results = {'Model': [], 'Vectorizer': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    for vectorizer, model in vm_combinations:

        if over_sampling:
            pipe = imb_pipeline([('vec', vectorizer),
                                 ('smote', SMOTE(random_state=random_state)),
                                 ('mod', model)])
        else:
            pipe = Pipeline([('vec', vectorizer),
                             ('mod', model)])

        print("-----------------------------------------------------------------------------------------")
        print(f"{turn_to_string(model)} with {turn_to_string(vectorizer)}")

        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        scorers = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        scores = cross_validate(pipe, df['preprocessed_text'], df['reference_label'],
                                scoring=scorers, cv=cv, n_jobs=-1, return_estimator=True)

        results['Model'].append(turn_to_string(model))
        results['Vectorizer'].append(turn_to_string(vectorizer))
        results['Accuracy'].append(np.mean(scores['test_accuracy']))
        results['Precision'].append(np.mean(scores['test_precision_weighted']))
        results['Recall'].append(np.mean(scores['test_recall_weighted']))
        results['F1 Score'].append(np.mean(scores['test_f1_weighted']))

        print("Model Complete")
        print("-----------------------------------------------------------------------------------------")

    return pd.DataFrame(results)
