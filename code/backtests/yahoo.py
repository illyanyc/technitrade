import yfinance as yf
from datetime import datetime, timedelta

def ohlcv(ticker: str,
         start_date,
         end_date):
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    return data