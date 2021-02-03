import pandas as pd
from sqlalchemy import create_engine
from yahooquery import Ticker

def read_api(tickes_list):

    try:
        ticker = Ticker(tickes_list)
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
    
    engine = create_engine('sqlite:///' + database_path)

    try:
        df.to_sql(table_name, engine, index=False, if_exists='replace')

    except Exception as e:
        print('Failed to save data in db: '+ str(e))

