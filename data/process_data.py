import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads and concatenates the messages and categories csv files into a pandas DataFrame.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left=messages, right=categories, on='id')
    return df

def clean_data(df):
    """
    Organize categories values and remove duplicates from final dataframe.
    """

    # organize catefories column
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0,:]
    categories.columns = [val[:-2] for val in row]

    for column in categories:
        # set each value to be the last character of the string
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: float(x.split('-')[1]))
    

    # drop old categories column from original dataframe
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filepath):
    engine = create_engine(database_filepath)
    df.to_sql('main_table', engine, index=False)


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