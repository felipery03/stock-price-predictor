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
    ''' A class to represent modeling pipeline, such as prep features,
    split dataset in train and test, train model and report results.

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
        ''' Preparate a dataframe with stock prices for a stock specified by
            symbol and market index for train period defined by start_date
            and end_date. In this dataset will be pad some days before and
            after train period to be possible calculate first features and last
            targets.

        params:
        stock_df (dataframe): Dataset with infos read from API
        symbol (string): Ticker of a specific stock
        start_date (datetime): First date from train period
        end_date (datetime): Last date from train period

        returns:
        stock (series): Data series with stock (symbol) price for
            train period + pad.
        ibov (series): Data series with market index price for
            train period + pad.
        '''
        # Pad dates
        start_date_pad = start_date - pd.DateOffset(days=180)
        end_date_pad = end_date + pd.DateOffset(days=45)
        
        # Filter only necessary info
        data = stocks_df[stocks_df['symbol'].isin([symbol, '^BVSP'])].copy()
        data = data[['symbol', 'date', 'adjclose']]
        
        # Filter date period
        stock = TrainModel.get_data(data, symbol, start_date_pad, end_date_pad)
        ibov = TrainModel.get_data(data, '^BVSP', start_date_pad, end_date_pad)
        
        # Only consider date in stock and ibov
        valid_dates = set(ibov.index).intersection(set(stock.index))
        
        # Sort series
        stock = stock.loc[valid_dates].sort_index()
        ibov = ibov.loc[valid_dates].sort_index()
        
        return (stock, ibov)

    @staticmethod
    def get_data(df, symbol, start_date, end_date):
        ''' Get data from df for a specif symbol in a specific period.

        params:
        df (dataframe): Input dataframe with info read from API.
        symbol (string): Ticker of a stock
        start_date (datetime): First date of the period.
        end_date (datetime): Last date of the period.

        returns:
        data (series): Data series with adjusted close price for the
            specific period.
        '''
        data = df[df.symbol == symbol].copy()
        
        data = data[(data.date >= start_date) & (data.date <= end_date)].copy()
        data.set_index('date', inplace=True)
        data = data.adjclose.rename(symbol)
        
        return data

    @staticmethod
    def prep_modeling_data(stock, ibov, targets_dict, start_date, end_date, target_flag):
        ''' Create features and targets for modeling. In this step, any date paddings
            are removed after features and targets creation.

        params:
        stock (series): Data series with adjclose price of a stock
        ibov (series): Data series with adjclose price of market index
        targets_dict (dictionary): Dictionary with keys as target specification and values
            as number of days to look a head. Example: key='1_day', value=1. This target 
            will be creating looking next day price from a date reference.
        start_date (datetime): First day of train period, if market is not open this day,
            it will be considered next one.
        end_date (datetime): Last day of train period, if market is not open this day,
            it will be considered last one.
        target_flag (boolean): Flag to control if modeling_df will return with calculated 
            targets (True) or only features (False).

        returns:
        modeling_df (dataframe): Dataframe with calculated features and targets based in 
            targets_dict (if target_flag=True) for the specific train set.
        '''
        # SMA for window 7, 14, 36, 73 and 146
        sma_7 = calculate_sma(stock, window=7)
        sma_14 = calculate_sma(stock, window=14)
        sma_36 = calculate_sma(stock, window=36)
        sma_73 = calculate_sma(stock, window=73)
        sma_146 = calculate_sma(stock, window=146) 
        
        # Momentum for window 36
        momentum = calculate_momentum(stock, window=36)
        
        # momentum SMA for window 7, 36 and 73. Window of momentum is
        # fixed in 36.
        sma_momentum_7 = calculate_sma(momentum, window=7)
        sma_momentum_36 = calculate_sma(momentum, window=36)
        sma_momentum_73 = calculate_sma(momentum, window=73)
        
        # SMA for market index with window 73
        sma_ibov_73 = calculate_sma(ibov, window=73)
        
        # Volatility for market index with window 73
        volatility_ibov = calculate_volatility(ibov, window=73)
        
        # Volatility for stock price with window 73
        volatility = calculate_volatility(stock, window=73)
        
        # Beta for stock x market index using 73 dates before date ref.
        beta = calculate_capm(stock, ibov, window=73).beta
        
        # EMA for window 7, 14, 36, 73 and 146
        ema_7 = calculate_ema(stock, window=7)
        ema_14 = calculate_ema(stock, window=14)
        ema_36 = calculate_ema(stock, window=36)
        ema_73 = calculate_ema(stock, window=73)
        ema_146 = calculate_ema(stock, window=146)
        
        # Putting all in a list more last stock price
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

            # Add target columns to vars list
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
    def train_test_split(modeling_df, target_period, test_size_ratio):
        ''' Split modeling_df in train and test. For test set will be used
            last days. Train set size is a pct of entire dataset defined by
            test_size_ratio. Target_period defines which target columns will
            be used.

        params:
        modeling_df (dataframe): Dataframe with features and targets for a data ref
            in each row.
        target_period (string): Target definition to be used based in targets_dict keys.
            Ex.: '1_day'
        test_size_ratio (float): Percentage of entire dataset used to test. This
            number should be 0 < test_size_ratio < 1.

        returns:
        X (dataframe): Entire dataframe with features
        y (series): Entire series with specific target
        X_train (dataframe): Train set with features
        X_test (dataframe): Test set with features
        y_train (series): Train series with specific target
        y_test (series): Test series with specific target
        '''
        # Calculate number of dates in test size
        test_size = round(modeling_df.shape[0] * test_size_ratio)
        
        # Split X and y
        X = modeling_df.filter(regex='^(?!target_)')
        y = modeling_df['target_' + target_period].copy()

        # Split train sets
        X_train = X.iloc[:-test_size]
        y_train = y.iloc[:-test_size]
        
        # split test_sets
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]

        return (X, y, X_train, X_test, y_train, y_test)

    @staticmethod
    def train_model(X, y, X_train, X_test, y_train, y_test, pipeline):
        ''' Train a specific pipeline using X_train and y_train. Calculate
            MAE for train and test set and refit pipeline with all data.

        params:
        X (dataframe): Entire dataframe with features
        y (series): Entire series with specific target
        X_train (dataframe): Train set with features
        X_test (dataframe): Test set with features
        y_train (series): Train series with specific target
        y_test (series): Test series with specific target
        pipeline (sklearn pipeline estimator): Pipeline to be fitted

        returns:
        result (dictionary): Dictionary with MAE from train and test set,
            and pipeline fitted with entire dataset.
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
        ''' Run all steps for extract features from stocks_df and fit all
            models for stocks in symbols and targets in targets_dict. Save results 
            in a pickle file in models_path.
        '''
        models = dict()

        # For each stock
        for symbol in self.symbols:
            # Prep data
            stock, ibov = self.prep_historical_data(
                    self.stocks_df,
                    symbol,
                    self.start_date,
                    self.end_date)

            # Prep vars to fit the model
            modeling_df = self.prep_modeling_data(
                    stock, 
                    ibov, 
                    self.targets_dict, 
                    self.start_date, 
                    self.end_date, 
                    target_flag=True)

            # Pipeline config (model choosen in Analysis)
            rf = RandomForestRegressor(random_state=0, n_estimators=1000)
            pipeline = make_pipeline(StandardScaler(), rf)
           
            # Train models
            results = dict()
            # For each target option
            for target_period in self.targets_dict.keys():
                 # Train test split, test 25% of dataset size
                X, y, X_train, X_test, y_train, y_test = self.train_test_split(
                        modeling_df,
                        target_period,
                        test_size_ratio=0.25)

                results[target_period] = self.train_model(
                        X,
                        y,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        pipeline)

            models[symbol] = results
        
        # Save models in a pickle file
        save_models(self.models_path, models)

    @staticmethod
    def calculate_results(vars_df, targets_dict, pipelines):
        ''' Using features in vars_df, fit some pipelines with different targets and
            compile the MAE test results in a dataframe.

        params:
        vars_df (dataframe): Features for a specific stock, each row contains a date ref
            in index
        targets_dict (dictionary): Dictionary with different targets varying how many
            days look ahead from a date ref
        pipelines (dictionary): Dictionary with different sklearn pipeline estimatores
            to analyze

        returns:
        results_df (dataframe): Dataframe with MAE test for some fitted pipelines with
            different targets
        '''
        results = []
        # For each target
        for target in targets_dict.keys():
            # Train pipeline with 25% of data for test set
            X, y, X_train, X_test, y_train, y_test = TrainModel.train_test_split(
                    vars_df,
                    target,
                    test_size_ratio=0.25)
            
            metrics = []
            metrics.append(target)
            # For each pipeline
            for _, pipeline in pipelines.items():
                # Fit model and return MAE test
                result_aux = TrainModel.train_model(
                        X,
                        y,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        pipeline)['mae_test']

                metrics.append(result_aux)
            results.append(metrics)
            
        # Convert results to a dataframe
        pipeline_names = list(pipelines.keys())
        results_df = pd.DataFrame(results, columns=['target'] + pipeline_names).T
        results_df.columns = results_df.iloc[0]
        results_df.drop(results_df.index[0], inplace=True)
        
        return results_df