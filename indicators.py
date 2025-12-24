import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data

# ref -> https://stackoverflow.com/questions/74283043/calculate-bollinger-band-using-pandas
# calculates Bollinger Bands and the Bollinger Band Percentage (%B) for the given price series
def bollinger_bands(prices, window=20, k = 2):
    # ref -> https://www.geeksforgeeks.org/python/how-to-calculate-moving-averages-in-python/
    simple_moving_avg = prices.rolling(window=window).mean()

    # ref -> https://www.statology.org/pandas-rolling-standard-deviation/
    rolling_std = prices.rolling(window=window).std()

    upper_band = simple_moving_avg + (k * rolling_std)
    lower_band = simple_moving_avg - (k * rolling_std)

    # ref -> https://www.tradingview.com/support/solutions/43000501971-bollinger-bands-b-b/
    bb_percentage = (prices - lower_band) / (upper_band - lower_band)

    return simple_moving_avg, upper_band, lower_band, bb_percentage

# ref -> https://www.macroption.com/rsi-calculation/#:~:text=RSI%20=%20100%20%E2%80%93%20100%20/%20(,and%20AvgD%20(see%20details%20below)
# computes the Relative Strength Index (RSI), a momentum oscillator ranging from 0–100
def relative_strength_index(prices, window=14):
    # ref -> https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
    delta = prices.diff()

    # separate gains and losses
    up_moves = delta.apply(lambda x: x if x > 0 else 0)
    down_moves = delta.apply(lambda x: abs(x) if x < 0 else 0)

    # ref -> https://www.geeksforgeeks.org/python/how-to-calculate-moving-averages-in-python/
    # calculate simple moving average (SMA) of gains and losses
    avg_gain = up_moves.rolling(window=window).mean()
    avg_loss = down_moves.rolling(window=window).mean()

    relative_strength = avg_gain / avg_loss

    relative_strength_index = 100 - (100 / (1 + relative_strength))

    return relative_strength_index

# ref -> https://www.investopedia.com/terms/s/stochasticoscillator.asp
# calculates the Stochastic Oscillator (%K and %D) to measure price momentum
def stochastic_oscillator(prices, k_window=14, d_window = 3):
    lowest = prices.rolling(window=k_window).min()
    highest = prices.rolling(window=k_window).max()

    k = 100 * ((prices - lowest) / (highest - lowest)) # fast stochastic indicator
    d = k.rolling(window=d_window).mean() # "slow" stochastic indicator; %D = 3-period moving average of %K

    return k, d

# ref -> https://www.investopedia.com/terms/m/macd.asp
# computes the MACD (Moving Average Convergence Divergence) indicator
def moving_average_convergence_divergence(prices, fast_window=12, slow_window=26):
    # ref -> https://www.geeksforgeeks.org/python/how-to-calculate-an-exponential-moving-average-in-python/
    fast_exponential_moving_avg = prices.ewm(span=fast_window, adjust=False).mean()
    slow_exponential_moving_avg = prices.ewm(span=slow_window, adjust=False).mean()

    # when positive, the short-term price is above the long-term trend -> bullish
    # when negative, short-term momentum is below the long-term trend -> bearish
    macd = fast_exponential_moving_avg - slow_exponential_moving_avg 

    # when MACD crosses above Signal -> potential buy signal
    # when MACD crosses below Signal -> potential sell signal
    signal = macd.ewm(span=9, adjust=False).mean()

    return macd, signal

# ref -> https://www.investopedia.com/terms/o/onbalancevolume.asp
# computes the On-Balance Volume (OBV) indicator to measure buying/selling pressure
def on_balance_volume(prices, volumes):
    # compute daily price differences
    delta = prices.diff()

    # determine “direction” of the move
    # ref -> https://www.askpython.com/python-modules/numpy/numpy-sign
    # ref -> https://stackoverflow.com/questions/2652368/how-to-detect-a-sign-change-for-elements-in-a-numpy-array
    direction = np.sign(delta)

    # multiply direction by volume to get signed volume
    signed_volume = direction * volumes

    # cumulatively sum signed volumes to form OBV series
    obv = signed_volume.cumsum()
    obv.iloc[0] = 0 # no trades on 0th day

    return obv
