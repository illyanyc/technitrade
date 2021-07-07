import yfinance as yf
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone('America/New_York')
dateformat = "%Y-%m-%d"

def ohlcv(ticker: str):
    date = datetime.now(IST)
    end_date = date.strftime(dateformat)
    start_date = date - timedelta(days=365)
    
    print(start_date, end_date)
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    return data