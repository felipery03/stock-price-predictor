import pandas as pd
from sqlalchemy import create_engine
import pickle
from yahooquery import Ticker

def read_api(tickers_list):
    ''' Read daily market data from yahooquery API for a set
        of stocks. It is API for yahoo finance data.

    params:
    tickers_list (list): List of Tickers stocks to be read.

    returns:
    stocks_df (dataframe): Return from API
    '''
    
    try:
        # Create API instance
        ticker = Ticker(tickers_list)
        # Get max period
        stocks_df = ticker.history(period="max")

    except Exception as e:
        print('Failed to read from API: '+ str(e))

    stocks_df.reset_index(inplace=True)
    
    return stocks_df

def save_data(df, database_path, table_name):
    ''' Save a dataframe in a sqlite db, creating a new table
    with table_name.

    Params:
    df (dataframe): Input dataframe
    database_path (string): Database path including database name
        and extension
    table_name (string): Table name which data will be inputed
    '''
    # Create engine
    engine = create_engine('sqlite:///' + database_path)

    try:
        df.to_sql(table_name, engine, index=False, if_exists='replace')

    except Exception as e:
        print('Failed to save data in db: '+ str(e))

def save_models(models_path, models):
    ' Save models in a pickle file'
    pickle.dump(models, open(models_path, 'wb'))

def load_models(models_path):
    ' Load models from a pickle file'
    models = pickle.load(open(models_path, 'rb'))
    
    return models