import sys
from move_data import read_api, save_data

class LoadData():
    ''' Class to intance a loader data from yahoo query API.
        It will read specific Tickers historical data, clean 
        and save in a db.

    Attributes:
    database_filepath (string): Location of db
    ticker_list (list): List of tickers to be imported
    '''

    def __init__(self, database_filepath):
        self.database_filepath = database_filepath
        self.ticker_list = ['^BVSP', 'ITSA4.SA', 'MGLU3.SA',
                'VVAR3.SA', 'WEGE3.SA', 'MDIA3.SA',
                'LREN3.SA', 'ITUB3.SA', 'EGIE3.SA']

    def clean_data(self, df):
        'Clean ticker string to remove ".SA".'
        data = df.copy()
        data.symbol = data.symbol.str.split('.').str[0]

        return data

    def load(self):
        ''' Run all steps to read data from API, clean data
            and save in a db.
        '''
        # Read data from API
        stocks_df = read_api(self.ticker_list)
        
        # Cleaning data
        clean_stocks_df = self.clean_data(stocks_df)

        # Save data in db
        save_data(clean_stocks_df, self.database_filepath, 'stocks')


