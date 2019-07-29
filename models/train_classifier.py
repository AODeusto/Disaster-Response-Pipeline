import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download(['punkt','stopwords','wordnet'])

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

def load_data(database_filepath):
    """
    Load data from the SQL db provided with the filepath
    INPUT
    database_filepath: path to the db
    OUTPUT
    X: df containing "message" col
    Y: df containing rest of the dataset
    category_names = list containing the name of each of the categories to classify
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Messages', engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    """
    Take a string and perform the following:
    - Normalize making it lowercase and removing punctuation
    - Tokenize
    - Remove Stop-Words
    - Stemming / Lemmatizing
    INPUT
    text: string containing text of the message
    OUTPUT
    lemmatized: Array of tokens after pocessing the text
    """
    # make every word lowercase 
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # tokenize text
    tokens = word_tokenize(text)    
    # lemmatize and remove stop words
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]    
    return lemmatized


def build_model():
    """
    INPUT 
    No input

    OUTPUT
    cv - GridSearchCV objedt containing pipeline and hyperparameters 
    in order to tune the model

    NOTES
    It builds a ML Pipeline including:
    - Vectorizer (Bag of words)
    - TFIDF Transformer
    - Multioutput Classifier
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight = 'balanced')))  ])
    print(pipeline.get_params())

    parameters = {'clf__estimator__n_estimators' : [40,100], 'clf__estimator__min_samples_split' : [2,3] }
    print ('Training pipeline in GridSearhCV')
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, scoring = 'f1_weighted', verbose = 3)
    
    return cv

def display_results(category_names, Y, y_test, y_pred):
    """
    INPUT:
    - category names  array of names for categories in the multioutput classifier
    - y_test  subset of data to test the model´s performance
    - y_pred  preds made by the model 

    OUTPUT
    - df containing model performance metrics: 'Accuracy', 'Precision', 'Recall', 'F1'
    """
    results = precision_recall_fscore_support(y_test, y_pred)
    metric = []
    for i, col in enumerate(category_names):
        accuracy = accuracy_score(y_test.iloc[:,i].values, y_pred[:,i])
        precision = precision_score(y_test.iloc[:,i].values, y_pred[:, i], average='weighted')
        recall = recall_score(y_test.iloc[:,i].values, y_pred[:, i], average='weighted')
        f1_sco = f1_score(y_test.iloc[:,i].values, y_pred[:, i], average='weighted')
        perc_df = Y[col].sum() / Y.shape[0]
        metric.append([accuracy, precision, recall, f1_sco, perc_df])
        
        
    # Create dataframe containing metrics
    metric = np.array(metric)
    metrics_df = pd.DataFrame(data = metric, index = category_names, 
                              columns = ['Accuracy', 'Precision', 'Recall', 'F1', '%df'])
      
    return metrics_df    

def evaluate_model(model, X_test, Y_test, Y, category_names):
    '''
    INPUT
    model: trained model 
    X_test: df containing test data excepting the label feature
    Y_test: df containing label feature for test data
    category_names: list of category names 
    OUTPUT
    metrics of the model based on real and predicted values
    '''
    # predict on test data
    y_pred = model.predict(X_test)
    metrics = display_results(category_names, Y, Y_test, y_pred)
    print(metrics)

def save_model(model, model_filepath):
    '''
    INPUT
    model: trained model
    model_filepath: path where to save the given trained model
    OUTPUT
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    '''
    Performs the whole training job using 
    the functions defined above
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, Y, category_names)

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