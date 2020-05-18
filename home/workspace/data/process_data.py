# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine 

def load_data(messages_filepath, categories_filepath):
    '''
    Function to read csv files for data import
    Input: filepaths for csv files
    Output: joined dataframe
    '''
    #load categories and messages csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how="inner", on="id")
    return df

def clean_data(df):
    '''
    Function to clean the dataframe
    Input: dataframe imported by load_data()
    Output: cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    columns = df.categories[1]
    columns = columns.replace('-', '')
    columns = columns.replace('0', '')
    columns = columns.replace('1', '')
    columns = columns.split(';')
    
    # rename the columns of `categories`
    filler = np.zeros(shape=(len(df),len(columns)))
    cats = df.categories.str.split(';')
    categories = pd.DataFrame.from_items(zip(cats.index, cats.values))
    categories = categories.transpose()
    categories.columns = columns

    #loop over categories leave last digit
    for i in range(0,len(categories.columns)):
        categories.iloc[:,i] = categories.iloc[:,i].str.slice(start=-1).astype(int)

     # drop the original categories column from `df`
    df.head()
    df=df.drop(['categories'], axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df= pd.concat([df.reset_index(drop=True), categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''
    Function to save cleaned dataframe to DB
    Input: cleaned dataframem, database path
    '''
    #create DB
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    #save to DB
    print(database_filename)
    df.to_sql('Dataset', engine, index=False)


def main():
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