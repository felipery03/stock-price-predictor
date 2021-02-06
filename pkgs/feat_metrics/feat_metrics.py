import pandas as pd
import numpy as np

def calculate_n_day_return(series, window=1):
    n_day_return = (series/series.shift(window)) - 1
    n_day_return.fillna(0, inplace=True)
    n_day_return = n_day_return.rename('day_' + str(window) + '_return')
    
    return n_day_return

def calculate_capm(series_a, series_b, window):
    arr_a = series_a.copy()
    arr_b = series_b.copy()

    if len(arr_a) != len(arr_b):
        raise ValueError('Length input arrays are different.')

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

        date = arr_w_a.tail(1).index[0]
        beta, alpha = np.polyfit(arr_w_a, arr_w_b, 1)
        result.append((date, beta, alpha))
        result_df = pd.DataFrame(result, columns=['date', 'beta', 'alpha'])
        result_df.set_index('date', inplace=True)
        result_df.sort_index(inplace=True)
        
    return result_df

def calculate_momentum(series, window):
    momentum = (series/series.shift(-window)) - 1
    momentum.fillna(0, inplace=True)
    momentum = momentum.rename('momentum')

    return momentum


def calculate_sma(series, window):
    sma = series.rolling(window).mean()
    sma_pct = (series/sma) - 1
    sma_pct.fillna(0, inplace=True)
    sma_pct = sma_pct.rename('sma_' + series.name + '_' + str(window))
    
    return sma_pct

def calculate_volatility(series, window):
    
    vol = series.rolling(window).std()
    vol.fillna(0, inplace=True)
    vol = vol.rename('volatility_' + series.name)

    return vol

def calculate_ema(series, window):
    
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
    target = series.shift(-n_days)
    target = target.rename('target_' + name)
    
    return target
