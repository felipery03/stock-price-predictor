import pandas as pd
import numpy as np

def calculate_n_day_return(series, window=1):
    ''' Calculate a percentage variation between two values in a time
        series. 

    params:
    series (series): Time series in a ascending order with float values.
    window (int): Quantity of rows shifted to make the calculation.

    returns:
    n_day_return (series): Series for each n day return calculated. For example: if window = 1, each
    row represents delta percentage between data ref in index and previous date.  
    '''
    n_day_return = (series/series.shift(window)) - 1
    n_day_return.fillna(0, inplace=True)
    n_day_return = n_day_return.rename('day_' + str(window) + '_return')
    
    return n_day_return

def calculate_capm(series_a, series_b, window):
    ''' Calculate CAPM indicator using daily return of series_a (specific stock) and 
    series_b (market index) in this equation: 
    daily_return_series_a = betha * daily_return_series_b + alpha.
    Window is used to limit data points used to fit this line based in data ref.

    params:
    series_a (series): calculate_n_day_return with window=1 for a specif stock.
    series_b (series): calculate_n_day_return with window=1 for ^BOVA (market index).
    window (int): Number of dates before data ref used to fit the line.

    returns:
    result_df (dataframe): Dataframe with alpha and beta calculated for each date usings 
        data points in a specif window.
    '''
    arr_a = series_a.copy()
    arr_b = series_b.copy()

    if len(arr_a) != len(arr_b):
        raise ValueError('Length input arrays are different.')

    # Calculate daily returns
    day_return_a = calculate_n_day_return(arr_a, 1)
    day_return_b = calculate_n_day_return(arr_b, 1)

    result = []

    # Slide the windown between the entire array
    # Last point was skipped because is not possible fit a line
    # only with 1 point
    for iter in range(0, len(arr_a) - 1):
        if iter == 0:
            arr_w_a = day_return_a[-window:]
            arr_w_b = day_return_b[-window:]
        else:
            arr_w_a = day_return_a[-(window + iter):-iter]
            arr_w_b = day_return_b[-(window + iter):-iter]

        # Get date ref
        date = arr_w_a.tail(1).index[0]
        
        # Fit the line
        beta, alpha = np.polyfit(arr_w_a, arr_w_b, 1)
        result.append((date, beta, alpha))

    # create dataframe
    result_df = pd.DataFrame(result, columns=['date', 'beta', 'alpha'])
    result_df.set_index('date', inplace=True)
    result_df.sort_index(inplace=True)
        
    return result_df

def calculate_momentum(series, window):
    ''' Calculate ratio between two points.

    params:
    series (series): Time series with float values.
    window (int): Number of dates shifted from date ref to calculate
    momentum between them. 

    returns:
    momentum (series): Series with momentum calculated for each date ref and
        a previous date determied by window param.
    '''
    momentum = (series/series.shift(-window)) - 1
    momentum.fillna(0, inplace=True)
    momentum = momentum.rename('momentum')

    return momentum

def calculate_sma(series, window):
    ''' Calculate simple move average using a data series
        in a determined window.

    params:
    series (series): Time series with float values.
    window (int): Previous number of dates used to calculation from
        data ref.  

    returns:
    sma_pct (series): SMA series calculated for each data ref in
        window period divided by data_ref value minus 1.
    '''
    sma = series.rolling(window).mean()
    sma_pct = (series/sma) - 1
    sma_pct.fillna(0, inplace=True)
    sma_pct = sma_pct.rename('sma_' + series.name + '_' + str(window))
    
    return sma_pct

def calculate_volatility(series, window):
    ''' Calculate standard deviation in a spefic window of dates in
        a series.

    params:
    series (series): Time series with float values.
    window (int): Previous number of dates used to calculation from
        data ref.  

    returns:
    vol (series): Volatility for each data ref calculated in a specific window
        of dates.
    '''
    vol = series.rolling(window).std()
    vol.fillna(0, inplace=True)
    vol = vol.rename('volatility_' + series.name)

    return vol

def calculate_ema(series, window):
    ''' Calculate Exponential mean average for a time series in a window
        of dates. It is similar with simple mean average, but recent dates 
        have more weight in the mean.

    params:
    series (series): Time series with float values.
    window (int): Previous number of dates used to calculation from
        data ref.  

    returns:
    ema_pct (series): EMA series calculated for each data ref in
        window period divided by data_ref value minus 1.

    '''
    arr = series.copy()
    
    result = []

    # Calculate last line
    result.append(arr[-window:].ewm(span=window, adjust=False).mean().tail(1))

    # Slide the windown between the entire array
    for iter in range(1, len(arr)):
        result.append(arr[-(window + iter):-iter].ewm(span=5).mean().tail(1))

    ema = pd.concat(result).sort_index()
    ema_pct = (arr/ema) - 1
    ema_pct.fillna(0, inplace=True)
    ema_pct = ema_pct.rename('ema_' + series.name + '_' + str(window))

    return ema_pct

def calculate_target(series, n_days, name):
    ''' Calculate shifted target for n_days in the future
        from a data ref.

    params:
    series:  Time series with float values.
    n_days (int): Number of workdays shifted for the future. 
    name (string): Target specification.

    returns:
    target (series): Target series renamed and shifted for its
    predict data ref. Ex.: if data ref is 2020-12-21 and n_days = 1, then
    the value of 2020-12-22 will be shifted for data ref.
    '''
    target = series.shift(-n_days).copy()
    target = target.rename('target_' + name)
    
    return target
