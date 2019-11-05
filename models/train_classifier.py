# import libraries
import sys
import warnings
warnings.filterwarnings(action="ignore")

import re
import pickle
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('select * from main_table', con=engine)
    df = df.drop('original', axis=1).dropna().sample(n=100)
    print(df.shape)

    X = df['message']
    Y = df.drop(['id', 'message', 'genre'], axis=1)

    return X, Y


def tokenize(text):
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word.lower().strip()) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Builds model Pipeline and set parameter for GridSearchCV.
    Return grid object (without fitting it to the data).
    """

    multi_clf = MultiOutputClassifier(XGBClassifier(objective='binary:logistic',
                                                    random_state=42))

    pipe = Pipeline([ # Feature Union (not currently being used)
            ('features', FeatureUnion([
                
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
            ])),

            ('clf', multi_clf)
        ])

    # parameters
    params = {'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
              'features__text_pipeline__vect__max_features': (None, 5000),
              'features__text_pipeline__tfidf__use_idf': (True, False),
              
              # XGBoost Classifier 
              'clf__estimator__n_estimators': [5],
              'clf__estimator__learning_rate': [1],
              'clf__estimator__max_depth':[5],
             }

    # grid Search
    grid = GridSearchCV(pipe, param_grid=params,
                        cv=2, n_jobs=1,
                        verbose=0, error_score=0,
                        return_train_score=True)

    return grid


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Get results and add them to a dataframe.
    def generate_classification_report(y_test, y_pred):
        classification_report_df = pd.DataFrame()
        
        # loop through categories and add results
        i=0
        for c in y_test.columns:
            precision, recall, f1_score, support = (precision_recall_fscore_support(
                y_test[c], y_pred[:,i], average='weighted'))
            
            # set values in dataframe
            classification_report_df.at[i, 'category' ] = c
            classification_report_df.at[i, 'f1_score'  ] = f1_score
            classification_report_df.at[i, 'precision'] = precision
            classification_report_df.at[i, 'recall'   ] = recall
            i+=1
        
        # print mean results
        print('Mean f_score:', classification_report_df['f1_score'].mean())
        print('Mean precision:', classification_report_df['precision'].mean())
        print('Mean recall:', classification_report_df['recall'].mean())
        
        # organize df for display
        classification_report_df.sort_values(by='f1_score', ascending=False, inplace=True)
        classification_report_df.set_index('category', inplace=True)
        
        return classification_report_df

    classification_report_df = generate_classification_report(y_test, y_pred)
    print(classification_report_df)


def save_model(model, model_filepath):
    """
    Save trained model.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()