import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import nltk
import pickle
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    '''
    Function to load the datza from a database
    Input: database_filepath
    Output: Data dependent & indepent features, names of dependent features
    '''
    # load data from database and transform data accordingly 
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('Dataset', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    return X, Y, Y.columns

def tokenize(text):
    '''
    Function to clean, tokenize & lemmatize sentences
    Input: text (sentence9)
    Output: tokenized sentence
    '''
    text= re.sub(r"[^a-zA-Z0-9]", " ",text.lower()) #normalize data
    tokens = word_tokenize(text) # tokenize data
    lemmatizer = WordNetLemmatizer() # initialize lemmatizer
    clean_tokens = []
    for tok in tokens: # lemmatize data
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model(X, Y, X_train, y_train):
    '''
    Function to build a model-pipeline for textmining and optimize it via gridsearch
    Input: independent & dependent features
    Output: initialized model-pipeline
    '''
    #create ML_Pipeline
    model_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters =  {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [30, 90], 
              'clf__estimator__min_samples_split': [2, 5]} 
    # perform gridsearch
    #cv = GridSearchCV(model_pipeline, param_grid=parameters, n_jobs=-1)
    #cv.fit(X_train, y_train)
    #return cv
    return model_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate the model_pipeline
    Input: model-pipeline, test set, true dependent features values of test-set
    Output: Classification performance of model
    '''
    
    y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Function to save the model_pipeline
    Input: model-pipeline & model_filepath 
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(database_filepath)
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X, Y, X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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