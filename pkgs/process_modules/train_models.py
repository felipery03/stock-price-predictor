from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import pandas as pd
from copy import deepcopy

from feat_metrics import calculate_n_day_return, calculate_sma, calculate_momentum
from feat_metrics import calculate_volatility, calculate_ema, calculate_capm, calculate_target
from move_data import save_models


class TrainModel():
    ''' A class to represent model interations.

    Attributes:
    stock_df (dataframe): Dataframe with stocks prices for specific tickers
        and ^BOVA index
    symbols (list): List of tickers that will be trained
    start_date (string): First date for train period, format (YYYY-MM-DD)
    end_date (string): Last date for train period, format (YYYY-MM-DD)
    models_path (string): Pathfile to save pickle models. 
    targets_dict (dict): Dictionary with defined target options in keys and
        how many workdays to look ahead in values.
    '''

    def __init__(self, stocks_df, symbols, start_date, end_date, models_path):
        ''' Constructs all the necessary attributes for TrainModel object.
        '''
        self.stocks_df = stocks_df.copy()
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.models_path = models_path
        self.targets_dict = {'1_day': 1,
                '1_week': 5,
                '2_weeks': 10,
                '1_month': 22}  

    @staticmethod
    def prep_historical_data(stocks_df, symbol, start_date, end_date):
        '''

        '''
        start_date_pad = start_date - pd.DateOffset(days=180)
        end_date_pad = end_date + pd.DateOffset(days=45)
        
        data = stocks_df[stocks_df['symbol'].isin([symbol, '^BVSP'])].copy()
        data = data[['symbol', 'date', 'adjclose']]
        
        stock = TrainModel.get_data(data, symbol, start_date_pad, end_date_pad)
        ibov = TrainModel.get_data(data, '^BVSP', start_date_pad, end_date_pad)
        
        valid_dates = set(ibov.index).intersection(set(stock.index))
        
        stock = stock.loc[valid_dates].sort_index()
        ibov = ibov.loc[valid_dates].sort_index()
        
        return (stock, ibov)

    @staticmethod
    def get_data(df, symbol, start_date, end_date):
        '''


        '''
        data = df[df.symbol == symbol].copy()
        
        data = data[(data.date >= start_date) & (data.date <= end_date)].copy()
        data.set_index('date', inplace=True)
        data = data.adjclose.rename(symbol)
        
        return data

    @staticmethod
    def prep_modeling_data(stock, ibov, targets_dict, start_date, end_date, target_flag):
        '''



        '''
        # SMA
        sma_7 = calculate_sma(stock, window=7)
        sma_14 = calculate_sma(stock, window=14)
        sma_36 = calculate_sma(stock, window=36)
        sma_73 = calculate_sma(stock, window=73)
        sma_146 = calculate_sma(stock, window=146) 
        
        momentum = calculate_momentum(stock, window=36)
        
        sma_momentum_7 = calculate_sma(momentum, window=7)
        sma_momentum_36 = calculate_sma(momentum, window=36)
        sma_momentum_73 = calculate_sma(momentum, window=73)
        
        sma_ibov_73 = calculate_sma(ibov, window=73)
        
        volatility_ibov = calculate_volatility(ibov, window=73)
        
        volatility = calculate_volatility(stock, window=73)
        
        beta = calculate_capm(stock, ibov, window=73).beta
        
        # EMA
        ema_7 = calculate_ema(stock, window=7)
        ema_14 = calculate_ema(stock, window=14)
        ema_36 = calculate_ema(stock, window=36)
        ema_73 = calculate_ema(stock, window=73)
        ema_146 = calculate_ema(stock, window=146)
        
        vars = [sma_7,
                sma_14,
                sma_36,
                sma_73,
                sma_146,
                sma_momentum_7,
                sma_momentum_36,
                sma_momentum_73,
                sma_ibov_73,
                volatility_ibov,
                volatility,
                beta,
                ema_7,
                ema_14,
                ema_36,
                ema_73,
                ema_146,
                stock]

        # If target_flag is True, include targets in dataframe
        if target_flag:
            targets = []

            # Create all possibles targets columns
            for target_label, target_days in targets_dict.items():
                target_aux = calculate_target(stock, n_days=target_days, name=target_label)
                targets.append(target_aux)

            vars = vars + targets

        # Putting all together
        modeling_df = pd.concat(vars, axis=1)
        
        modeling_df.reset_index(inplace=True)
        modeling_df.rename({'index': 'date'}, axis=1, inplace=True)
        
        # Only train with days inside of choosen period
        train_period = (modeling_df.date >= start_date) & (modeling_df.date <= end_date)
        modeling_df = modeling_df[train_period].copy()
        modeling_df.set_index('date', inplace=True)
        
        return modeling_df

    @staticmethod
    '''



    '''
    def train_test_split(modeling_df, target_period, test_size_ratio):
        test_size = round(modeling_df.shape[0] * test_size_ratio)
        
        X = modeling_df.filter(regex='^(?!target_)')
        y = modeling_df['target_' + target_period].copy()

        # Drop targets
        X_train = X.iloc[:-test_size]
        y_train = y.iloc[:-test_size]
        
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]

        return (X, y, X_train, X_test, y_train, y_test)

    @staticmethod
    def train_model(X, y, X_train, X_test, y_train, y_test, pipeline):
        '''



        '''
        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        # Refit
        pipeline.fit(X, y)
        
        result = {'mae_train': mae_train,
                'mae_test': mae_test,
                'model': deepcopy(pipeline)}
        
        return result    


    def create_models(self):
        '''



        '''
        models = dict()

        for symbol in self.symbols:
            # Prep data
            stock, ibov = self.prep_historical_data(self.stocks_df, symbol, self.start_date, self.end_date)

            # Prep vars to fit the model
            modeling_df = self.prep_modeling_data(stock, ibov, self.targets_dict, self.start_date, self.end_date, True)

            # Pipeline config
            rf = RandomForestRegressor(random_state=0, n_estimators=1000)
            pipeline = make_pipeline(StandardScaler(), rf)
           
            # Train models
            results = dict()
            for target_period in self.targets_dict.keys():
                 # Train test split, test 25% of dataset size
                X, y, X_train, X_test, y_train, y_test = self.train_test_split(modeling_df, target_period, 0.25)

                results[target_period] = self.train_model(X, y, X_train, X_test, y_train, y_test, pipeline)

            models[symbol] = results
        
        # Save models in a pickle file
        save_models(self.models_path, models)

    # def prep_predict_data(self, forecast_date, symbol):
    #     stock, ibov = self.prep_historical_data(self.stocks_df,
    #             symbol,
    #             start_date=forecast_date,
    #             end_date=forecast_date)

    #     modeling_df = self.prep_modeling_data(stock,
    #             ibov,
    #             self.targets_dict,
    #             start_date=forecast_date,
    #             end_date=forecast_date,
    #             target_flag=False)
        
    #     return modeling_df

    @staticmethod
    def calculate_results(vars_df, targets_dict, pipelines):
        '''


        '''
        results = []
        for target in targets_dict.keys():
            X, y, X_train, X_test, y_train, y_test = TrainModel.train_test_split(vars_df, target, 0.25)
            
            metrics = []
            metrics.append(target)
            for pipeline_name, pipeline in pipelines.items():
                result_aux = TrainModel.train_model(X, y, X_train, X_test, y_train, y_test, pipeline)['mae_test']
                metrics.append(result_aux)
            results.append(metrics)
            
        # Convert results to a dataframe
        pipeline_names = list(pipelines.keys())
        results_df = pd.DataFrame(results, columns=['target'] + pipeline_names).T
        results_df.columns = results_df.iloc[0]
        results_df.drop(results_df.index[0], inplace=True)
        
        return results_df