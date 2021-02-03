import sys
from utils import read_api, save_data

def main():
    if len(sys.argv) == 2:

        # Get db filepath
        database_filepath = sys.argv[1]

        # Ticker stocks choosen
        tickers_list = ['^BVSP', 'ITSA4.SA', 'MGLU3.SA',
                'VVAR3.SA', 'WEGE3.SA', 'MDIA3.SA',
                'LREN3.SA', 'ITUB3.SA', 'EGIE3.SA']

        print('Reading data from API...')
        stocks_df = read_api(tickers_list)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(stocks_df, database_filepath, 'stocks')

    else:
        print('''Please provide the filepath of the database 
        to save data collected from API as first argument.
        Example: python load_data.py data/Stocks.db''')

if __name__ == '__main__':
    main()