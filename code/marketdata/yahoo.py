import yfinance as yf
from datetime import datetime, timedelta
from pandas import DataFrame, concat
import pytz
import time

IST = pytz.timezone('America/New_York')
dateformat = "%Y-%m-%d"

def ohlcv(tickers: list or DataFrame, 
          start_date : str = '2020-01-01',
          end_date : str = datetime.now().strftime('%Y-%m-%d')):
    
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
    
    Returns
    -------
    ohlcv_df : DataFrame with securities price data
    '''
    
    ohlcv = {}
    tickers_n = 50
    
    for ticker in tickers:
        
        df = yf.download(ticker, start=start_date, end=end_date)

        ohlcv[ticker] = df
        
        time.sleep(0.1)
        
    ohlcv_df = concat(ohlcv.values(), keys=ohlcv.keys())
        
    return ohlcv_df
