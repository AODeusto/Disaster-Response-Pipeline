# Disaster Response Pipeline Project

### Content

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)
6. [Instructions](#instructions)

## Installation <a name="installation"></a>

*eyond the Anaconda distribution of Python, the following packages need to be installed for nltk:
* Anaconda installation for Python: pandas, scikitlearn, numpy
* punkt, wordnet, stopwords packages for NLTK


## Project Motivation<a name="motivation"></a>

By using data engineering techniques we can preprocess messages converting them into tokenized words using NLP and train a Machine Learning multioutput classificator in order to build an API that classifies messages in a disaster scenario. This classification could be a great help to agencies or NGO´s automatizing the process of setting priorities in the messages they receive.


## File Descriptions <a name="files"></a>

There are three main foleders:
1. data
    - disaster_categories.csv: dataset with all the categories.
    - disaster_messages.csv: dataset with all the messages.
    - process_data.py: ETL pipeline for loading, cleaning, and saving data into a SQL database.
    - DisasterResponse.db: output of the ETL pipeline.
2. models
    - train_classifier.py: machine learning pipeline to train an save a classifier from database previously created with process_data.py
    - classifier.pkl: output of the machine learning pipeline.
3. app
    - run.py: File for building a flask web app showing visualizations about the analysis.
    - templates folde contains html code for the web app.

## Results<a name="results"></a>

1. ETL pipleline to load data from two csv files, clean data, and save it to a SQLite DB
2. Machine Learning pipepline train a Multioutput Classifier on the categories included into the dataset.
3. Flask app showing data visualizations and classifying a message introduced by user.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits must be given to Udacity for the starter codes and FigureEight for provding the data used by this project. 

## Instructions:<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
