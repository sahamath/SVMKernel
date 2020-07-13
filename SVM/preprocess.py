import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

import technical_indicators as ta


def load_csv(path):
    """
		Loads csv file and del 'Date' and 'Adj Close' columns
		
		Args:
		path: path of the csv file (index of stocks) to load 
	"""

    aapl = pd.read_csv(path)
    del aapl["Date"]
    if "Adj Close" in aapl.columns:
        del aapl["Adj Close"]
    return aapl


def get_exp_preprocessing(data_f, alpha=0.9):
    """
		Returns the smoothened dataframe (exponentially weighted moving average)
		Args:
		df: the pandas dataframe
		alpha: smoothing factor
	"""
    df = data_f.copy()
    edata = df.ewm(alpha=alpha).mean()
    return edata


def feature_extraction(data_arg, trading_days):
    """
		Extracts the important features necessary for classification
	"""
    data = data_arg.copy()
    print("Trading Days:" + str(trading_days))
    for x in trading_days:
        data = ta.relative_strength_index(data, n=x)
        data = ta.stochastic_oscillator_d(data, n=x)
        data = ta.accumulation_distribution(data, n=x)
        data = ta.average_true_range(data, n=x)
        data = ta.momentum(data, n=x)
        data = ta.money_flow_index(data, n=x)
        data = ta.rate_of_change(data, n=x)
        data = ta.on_balance_volume(data, n=x)
        data = ta.commodity_channel_index(data, n=x)
        data = ta.ease_of_movement(data, n=x)
        data = ta.trix(data, n=x)
        data = ta.vortex_indicator(data, n=x)

    data["ema50"] = data["Close"] / data["Close"].ewm(50).mean()
    data["ema21"] = data["Close"] / data["Close"].ewm(21).mean()
    data["ema14"] = data["Close"] / data["Close"].ewm(14).mean()
    data["ema5"] = data["Close"] / data["Close"].ewm(5).mean()

    data = ta.macd(data, n_fast=12, n_slow=26)

    del data["Open"]
    del data["High"]
    del data["Low"]
    del data["Volume"]

    return data


def compute_prediction_int(df, n):
    """
		Computes the labels
	"""
    pred = df.shift(-n)["Close"] >= df["Close"]
    pred = pred.iloc[:-n]
    return pred.astype(int)


def prepare_data(data_f, horizon, alpha=0.9, trading_days=3):

    aapl = data_f.copy()
    saapl = get_exp_preprocessing(aapl, alpha)
    data = feature_extraction(saapl, trading_days).dropna().iloc[:-horizon]
    data["pred"] = compute_prediction_int(data, n=horizon)
    del data["Close"]
    return data.dropna()
