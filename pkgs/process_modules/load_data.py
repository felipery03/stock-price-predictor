import sys
from move_data import read_api, save_data

class LoadData():
    def __init__(self, database_filepath):
        self.database_filepath = database_filepath
        self.ticker_list = ['^BVSP', 'ITSA4.SA', 'MGLU3.SA',
                'VVAR3.SA', 'WEGE3.SA', 'MDIA3.SA',
                'LREN3.SA', 'ITUB3.SA', 'EGIE3.SA']

    def clean_data(self, df):

        data = df.copy()
        data.symbol = data.symbol.str.split('.').str[0]

        return data

    def load(self):
        stocks_df = read_api(self.ticker_list)
        
        # Cleaning data
        clean_stocks_df = self.clean_data(stocks_df)

        # Save data in db
        save_data(clean_stocks_df, self.database_filepath, 'stocks')


