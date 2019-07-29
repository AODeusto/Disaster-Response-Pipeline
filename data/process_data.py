import sys

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# df = load_data(messages_filepath,categories_filepath)

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath: path to file containing the 
                       csv file with messages data
    categories_filepath: path to file containing the 
                         csv file with categories data and names
    OUTPUT
    df: merged dataframe containig data from both csv files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'left', on = ['id'])
    return df


def clean_data(df):
    '''
    INPUT
    df: dataframe with data to be cleaned 
    OUTPUT
    df: cleaned dataframe with features engineered and hot-encoded
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = [i.split('-')[0] for i in row]
    print('Category names are ', category_colnames)
    # Name columns in 'categories' as category_colnames
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    print('Labels cleaned')
    # In order to pass the input into a classifier, we need to check if values in 
    # categories are just 0's and 1's. Also store those columns where there are any other 
    # apart from 1s or 0s
    to_delete = []
    other_values= []
    for column in categories:
        if len(np.unique(categories[column])) == 1:
            to_delete.append(column)
        elif len(np.unique(categories[column])) > 2:
            other_values.append(column)
    print ('Columns {} just have 0s or 1s'.format(to_delete))
    # Drop columns contained in the list in to_delete
    categories.drop(to_delete, axis = 1, inplace = True)
    print ('Columns dropped')
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    print ('Dropped the original categories column')
    # concatenate the original dataframe with the new `categories` dataframe
    print ('Concatenating original df with new separated categories df')
    df = pd.concat([df, categories], axis = 1)  
    # Drop rows containing other values nor 1 nor 0
    print ('Columns {} contain other values apart from 0 or 1'.format(other_values))
    for col in other_values:
        df = df[(df[col] == 0) | (df[col] == 1)]  
    # drop duplicates
    print ('Dropping duplicates')
    df.drop_duplicates(inplace = True)
    return df

def save_data(df, database_filename):
    '''
    INPUT
    model: trained model
    model_filepath: path where to save the given trained model
    OUTPUT
    '''
    table_n = 'Messages'
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(table_n, engine, index=False, if_exists='replace')
      
def main():
    '''
    Performs the whole data processing based
    on functions defined above
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()