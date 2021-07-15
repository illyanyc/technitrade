# System libraries
import os, time, sys, glob
from pathlib import Path
from datetime import date, datetime, timedelta

# API libraries
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Data libraries
import pandas as pd
from pandas import DataFrame, Timestamp, concat

alpaca_secret_key = str()
alpaca_secret_key = str()


def load_api_keys(api_path: str = 'api_keys.env',
                  debug: bool = False):
    '''Instantiates connection to Alpaca Trade API'''

    # load API keys
    load_dotenv(api_path)

    # set Alpaca API key and secret
    alpaca_api_key = 'PKPVNYCI99P4BO4UR76W'
    alpaca_secret_key = 'njpB50wHS3xfmFqFutenCqDIcSm6L665BXs5wCc8'

    if debug:
        print(f"Testing Apaca Trade API key by data type:")
        print(f"ALPACA_API_KEY: {type(alpaca_api_key)}")
        print(f"ALPACA_SECRET_KEY: {type(alpaca_secret_key)}")

    # create the Alpaca API object
    api = tradeapi.REST(
        alpaca_api_key,
        alpaca_secret_key,
        api_version="v2"
    )

    return api


def ohlcv(tickers: list or DataFrame,
          start_date: str = '2020-01-01',
          end_date: str = datetime.now().strftime('%Y-%m-%d'),
          timeframe: str = '1D',
          api_key_path: str = 'api_keys.env',
          debug: bool = False
          ) -> DataFrame:
    '''Returns pd.DataFrame with prices for the given tickers

    ...

    Parameters
    ----------
    tickers : list of str
        list of tickers
    start_date : str
        string with date in following format YYYY-MM-DD; default = '2020-01-01'
    end_date : str
        string with date in following format YYYY-MM-DD; default = today's date {datetime.now.strftime('%Y-%m-%d')}
    timeframe : str
        timeframe for the ohlcv barset; default = '1D'. The valid intervals are: 1Min, 5Min, 15Min and 1D
    api_key_path : str
        path for the .env file containing Alpaca Trade API key and secret key


    Returns
    -------
    ohlcv_df : DataFrame with securities price data
    '''
    api = load_api_keys(api_key_path, debug=debug)

    # parse start and end dates
    start_date = Timestamp(start_date, tz="America/New_York").isoformat()
    end_date = Timestamp(end_date, tz="America/New_York").isoformat()

    # connect to Alpaca Trade API and get ohlcv
    """Condition handling: Alpaca API 422 Client Error if more than 100 tickers are passed"""
    ohlcv_df = DataFrame()
    tickers_n = 50

    for i in range(0, len(tickers), tickers_n):
        sliced_tickers = tickers[i:i + tickers_n]

        df = api.get_barset(
            sliced_tickers,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            limit=1000).df

        ohlcv_df = concat([ohlcv_df, df], axis="columns", join="outer")
        time.sleep(0.1)

    return ohlcv_df


def test():
    pass


def main():
    pass


if __name__ == '__main__':
    main()