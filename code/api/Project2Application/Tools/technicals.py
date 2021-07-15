'''TODO: documentation'''

# Import libraries
# System
import os, time, sys, glob
from pathlib import Path
from datetime import date, datetime, timedelta

# Data
import numpy as np
import pandas as pd
from pandas import DataFrame


class TechnicalAnalysis:
    '''
    Class used to calculate technical indicators

        Attributes
        ----------
        ohlcv : DataFrame
            a multiindexed DataFrame of Open, High, Low, Close, Volume data of Tickers

    '''

    def __init__(self, _ohlcv):
        '''
        Parameters
        ----------
        _ohlcv : DataFrame
            current working multiindexed DataFrame of Open, High, Low, Close, Volume data of Tickers
        '''
        self.ohlcv = _ohlcv

    def validate_ticker(self,
                        ticker: str):
        '''Helper method - validates if ticker is not null, not empty, and is present in the self.ohlcv'''

        _tickers = self.tickers()

        if not ticker:
            raise ValueError(f"Ticker param is empty: please pass a correct ticker -> ticker = str({ticker})")
        elif ticker:
            if isinstance(ticker, str):
                if ticker not in _tickers:
                    raise ValueError(f"Ticker {ticker} not found in self.ohlcv DataFrame")
                elif ticker in _tickers:
                    pass

            elif isinstance(ticker, int):
                raise TypeError(f"Incorrect ticker format: ticker cannot contain integers")
            else:
                raise TypeError(f"Incorrect data type: ticker must be a str -> str({ticker})")

    def tickers(self):
        '''Returns a list of tickers inside ohlcv
        '''
        df = self.ohlcv
        return list(df.columns.levels[0])

    def _open(self,
              ticker) -> DataFrame:
        '''
        Returns open price for ticker

            Parameters
            ----------
            ticker : str
                ticker to be processed

            Returns
            -------
            df : DataFrame
                'close' values
        '''
        self.validate_ticker(ticker=ticker)  # validate ticker

        df = self.ohlcv.xs('open',
                           axis=1,
                           level=1,
                           drop_level=False).droplevel(1, axis=1)

        return DataFrame(df[ticker]).rename(columns={ticker: 'open'})

    def _high(self,
              ticker) -> DataFrame:
        '''
        Returns high price for ticker

            Parameters
            ----------
            ticker : str
                ticker to be processed

            Returns
            -------
            df : DataFrame
                'high' values
        '''
        self.validate_ticker(ticker=ticker)  # validate ticker

        df = self.ohlcv.xs('high',
                           axis=1,
                           level=1,
                           drop_level=False).droplevel(1, axis=1)

        return DataFrame(df[ticker]).rename(columns={ticker: 'high'})

    def _low(self,
             ticker) -> DataFrame:
        '''
        Returns low price for ticker

            Parameters
            ----------
            ticker : str
                ticker to be processed

            Returns
            -------
            df : DataFrame
                'low' values
        '''
        self.validate_ticker(ticker=ticker)  # validate ticker

        df = self.ohlcv.xs('low',
                           axis=1,
                           level=1,
                           drop_level=False).droplevel(1, axis=1)

        return DataFrame(df[ticker]).rename(columns={ticker: 'low'})

    def _close(self,
               ticker) -> DataFrame:
        '''
        Returns close price for ticker

            Parameters
            ----------
            ticker : str
                ticker to be processed - default = 'AAPL'

            Returns
            -------
            df : DataFrame
                'close' values
        '''
        self.validate_ticker(ticker=ticker)  # validate ticker

        df = self.ohlcv.xs('close',
                           axis=1,
                           level=1,
                           drop_level=False).droplevel(1, axis=1)

        return DataFrame(df[ticker]).rename(columns={ticker: 'close'})

    def _volume(self,
                ticker) -> DataFrame:
        '''
        Returns volume for ticker

            Parameters
            ----------
            ticker : str
                ticker to be processed

            Returns
            -------
            df : DataFrame
                'volume' values
        '''
        self.validate_ticker(ticker=ticker)  # validate ticker

        df = self.ohlcv.xs('volume',
                           axis=1,
                           level=1,
                           drop_level=False).droplevel(1, axis=1)

        return DataFrame(df[ticker]).rename(columns={ticker: 'volume'})

    # RSI Values
    def rsi(self,
            ticker: str,
            days: int = 14) -> DataFrame:
        '''
        Returns pd.DataFrame with RSI values

            Parameters
            ----------
            days : int
                number of days for RSI calculation; default = 14
            ticker : str
                ticker to be processed

            Returns
            -------
            rsi : DataFrame
                RSI values
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        close = self.ohlcv.loc[:, ticker]['close'].copy()

        # calculate delta
        delta = close.diff()

        # calculate gain and loss
        n_up = delta.clip(lower=0)
        n_down = -1 * delta.clip(upper=0)

        # calculate ema
        ema_up = n_up.ewm(com=days, adjust=False).mean()
        ema_down = n_down.ewm(com=days, adjust=False).mean()

        # calculate relative strenght
        rs = ema_up / ema_down

        # calculate rsi and append to close
        rsi = 100 - (100 / (1 + rs))

        del close

        return DataFrame(rsi.fillna(0)).rename(columns={'close': 'rsi'})

    # Williams %R Values
    def williams(self,
                 ticker: str,
                 days: int = 14) -> DataFrame:
        '''
        Returns pd.DataFrame with Williams %R values

            Parameters
            ----------
            days : int
                number of days for RSI calculation; default = 14
            ticker : str
                ticker to be processed

            Returns
            -------
            williams_range : DataFrame
                Williams %R values
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        hlc = self.ohlcv.loc[:, ticker][['high', 'low', 'close']].copy()

        highest_high = hlc['high'].rolling(window=days, center=False).max()
        lowest_low = hlc['low'].rolling(window=days, center=False).min()
        williams_range = (-100) * ((highest_high - hlc['close']) / (highest_high - lowest_low))

        del hlc

        return DataFrame(williams_range.fillna(0)).rename(columns={0: 'williams_range'})

    # Aroon Indicator
    def aroon(self,
              ticker: str,
              days: int = 14) -> DataFrame:

        '''
        Returns pd.DataFrame with Aroon Oscillator values

            Parameters
            ----------
            days : int
                number of days for Aroon Oscillator calculation; default = 14
            ticker : str
                ticker to be processed

            Returns
            -------
            aroon : DataFrame
                Aroon Oscillator
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        close = self.ohlcv.loc[:, ticker]['close'].copy()

        aroon_up = []
        aroon_down = []

        n_days = days
        while n_days < len(list(close.index)):
            date = close[n_days - days:n_days].index

            up = ((close[n_days - days:n_days].tolist().index(max(close[n_days - days:n_days]))) / days) * 100
            aroon_up.append(up)

            down = ((close[n_days - days:n_days].tolist().index(min(close[n_days - days:n_days]))) / days) * 100
            aroon_down.append(down)

            n_days += 1

        aroon = DataFrame([0] * days + [float(au - ad) for au, ad in zip(aroon_up, aroon_down)], index=df.index)

        del close

        return aroon.fillna(0).rename(columns={0: 'aroon'})

    # Money Flow Index
    def mfi(self,
            ticker: str,
            days: int = 14) -> DataFrame:
        '''
        Returns pd.DataFrame with Money Flow Index values

            Parameters
            ----------
            days : int
                number of days for Money Flow Index calculation; default = 14
            ticker : str
                ticker to be processed

            Returns
            -------
            mfi : DataFrame
                Money Flow Index
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        hlc = self.ohlcv.loc[:, ticker][['high', 'low', 'close', 'volume']].copy()

        # typical price
        hlc['tp'] = (hlc['high'] + hlc['low'] + hlc['close']) / 3

        # raw money flow
        hlc['rmf'] = hlc['tp'] * hlc['volume']

        # positive and negative money flow
        hlc['pmf'] = np.where(hlc['tp'] > hlc['tp'].shift(1), hlc['tp'], 0)
        hlc['nmf'] = np.where(hlc['tp'] < hlc['tp'].shift(1), hlc['tp'], 0)

        hlc['mfr'] = hlc['pmf'].rolling(window=14, center=False).sum() / hlc['nmf'].rolling(window=days,
                                                                                            center=False).sum()

        mfi = 100 - 100 / (1 + hlc['mfr'])

        del hlc

        return DataFrame(mfi.fillna(0)).rename(columns={0: 'mfi'})

    # Stoichastic Oscillator
    def stoch(self,
              ticker: str,
              days: int = 14) -> DataFrame:
        '''
        Returns pd.DataFrame with Stochastic Oscillator values

            Parameters
            ----------
            days : int
                number of days for Stochastic Oscillator calculation; default = 14
            ticker : str
                ticker to be processed

            Returns
            -------
            stoch : DataFrame
                Stochastic Oscillator
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        hlc = self.ohlcv.loc[:, ticker][['high', 'low', 'close']].copy()

        stoch_k = ((hlc['close'] - hlc['low'].rolling(window=days, center=False).mean()) / (
                    hlc['high'].rolling(window=days, center=False).max() - hlc['low'].rolling(window=days,
                                                                                              center=False).min())) * 100

        #         stoch_d = stoch_k.rolling(window = days-1, center=False).mean()

        del hlc

        return DataFrame(stoch_k.fillna(0)).rename(columns={0: 'stoch_k'})

    # Price Volume Trend
    def pvt(self,
            ticker: str,
            days: int = 14) -> DataFrame:
        '''
        Returns pd.DataFrame with Price Volume Trend values

            Parameters
            ----------
            days : int
                number of days for Stochastic Oscillator calculation; default = 14
            ticker : str
                ticker to be processed

            Returns
            -------
            pvt : DataFrame
                Prive Volume Trend
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        hlc = self.ohlcv.loc[:, ticker][['close', 'volume']].copy()

        momentum = (hlc['close'] - hlc['close'].shift(1)).fillna(0)
        pvt = (momentum / hlc['close'].shift(1)) * hlc['volume']
        pvt = pvt - pvt.shift(1)

        del hlc

        return DataFrame(pvt.fillna(0)).rename(columns={0: 'pvt'})

    # Moving Average Convergence Divergence
    def macd(self,
             ticker: str,
             period: tuple = (26, 12)) -> DataFrame:
        '''
        Returns pd.DataFrame with Moving Average Convergence Divergence values

            Parameters
            ----------
            period : tuple
                number of days for MACD calculation; default = (26,12)
            ticker : str
                ticker to be processed

            Returns
            -------
            macd : DataFrame
                MACD
        '''
        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        close = self.ohlcv.loc[:, ticker]['close'].copy()

        ema_period0 = close.ewm(span=period[0]).mean().fillna(0)
        ema_period1 = close.ewm(span=period[1]).mean().fillna(0)

        macd = (ema_period1 - ema_period0)

        del close

        return DataFrame(macd.fillna(0)).rename(columns={0: 'macd'})

    # Moving Average
    def ma(self,
           ticker: str,
           days: int) -> DataFrame:
        '''
        Returns pd.DataFrame with Moving Average values

            Parameters
            ----------
            days : int
                number of days for Moving Average calculation
            ticker : str
                ticker to be processed

            Returns
            -------
            ma : DataFrame
                Moving Average
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        close = self.ohlcv.loc[:, ticker]['close'].copy()

        ma = close.shift(1).rolling(window=days).mean().fillna(method='ffill')

        del close

        return DataFrame(ma.fillna(0)).rename(columns={0: str('ma_' + str(days))})

    # Exponential Moving Average
    def ema(self,
            ticker: str,
            days: int) -> DataFrame:
        '''
        Returns pd.DataFrame with Exponential Moving Average values

            Parameters
            ----------
            days : int
                number of days for Exponential Moving Average calculation
            ticker : str
                ticker to be processed

            Returns
            -------
            ema : DataFrame
                Exponential Moving Average
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        close = self.ohlcv.loc[:, ticker]['close'].copy()

        ema = close.ewm(span=days).mean().fillna(0)

        del close

        return DataFrame(ema.fillna(0)).rename(columns={0: str('ma_' + str(days))})

    # High Low
    def highlow(self,
                ticker: str) -> DataFrame:
        '''
        Returns pd.DataFrame with High - Low difference values

            Parameters
            ----------
            ticker : str
                ticker to be processed

            Returns
            -------
            highlow : DataFrame
                HighLow
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        hl = self.ohlcv.loc[:, ticker][['high', 'low']].copy()

        highlow = hl['high'] - hl['low']

        del hl

        return DataFrame(highlow.fillna(0)).rename(columns={0: str('highlow')})

    # Close Open
    def closeopen(self,
                  ticker: str) -> DataFrame:
        '''
        Returns pd.DataFrame with Close - Open difference values

            Parameters
            ----------
            ticker : str
                ticker to be processed

            Returns
            -------
            closeopen : DataFrame
                CloseOpen
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        hl = self.ohlcv.loc[:, ticker][['close', 'open']].copy()

        highlow = hl['close'] - hl['open']

        del hl

        return DataFrame(highlow.fillna(0)).rename(columns={0: str('closeopen')})

    # Bollinger Bands
    def bollingerbands(self,
                       ticker: str,
                       window: str = 21,
                       stdevs: int = 2) -> DataFrame:
        '''
        Returns pd.DataFrame with Close - Open difference values

            Parameters
            ----------
            ticker : str
                ticker to be processed
            window : str
                time window for Bollinger Band calculation
            stdevs : str
                number of standard deviations

            Returns
            -------
            bb : DataFrame
                Bollinger Band High, Bollinger Band Low
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        close = self.ohlcv.loc[:, ticker][['close']].copy()

        rolling_mean = close.rolling(window).mean()
        rolling_stdev = close.rolling(window).std()

        bb_high = (rolling_mean + (rolling_stdev * stdevs)).fillna(method='ffill')
        bb_low = (rolling_mean - (rolling_stdev * stdevs)).fillna(method='ffill')

        del close

        return DataFrame(bb_high.fillna(0)).rename(columns={0: str('bb_high')}), DataFrame(bb_low.fillna(0)).rename(
            columns={0: str('bb_low')})

    # Daily Returns
    def daily_return(self,
                     ticker: str) -> DataFrame:

        '''
        Returns pd.DataFrame with Daily Return values

            Parameters
            ----------
            ticker : str
                ticker to be processed

            Returns
            -------
            daily_return : DataFrame
                Daily returns as .pct_change()
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        close = self.ohlcv.loc[:, ticker][['close']].copy()

        daily_return = close.pct_change().fillna(0)

        del close

        return DataFrame(daily_return.fillna(0)).rename(columns={0: str('daily_return')})

    # Cumulative Daily Returns
    def cum_daily_return(self,
                         ticker: str) -> DataFrame:

        '''
        Returns pd.DataFrame with Cumulative Daily Return values

            Parameters
            ----------
            ticker : str
                ticker to be processed

            Returns
            -------
            cum_daily_return : DataFrame
                Daily returns as .cumprod()
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        close = self.ohlcv.loc[:, ticker][['close']].copy()

        daily_return = close.pct_change().fillna(0)
        cum_daily_return = (1 + daily_return).cumprod()

        del close

        return DataFrame(cum_daily_return.fillna(0)).rename(columns={0: str('cum_daily_return')})

    # Lagging Returns
    def lagging_returns(self,
                        ticker: str,
                        days: int = 7) -> DataFrame():
        '''
        Returns pd.DataFrame with all Technical Indicators

            Parameters
            ----------
            ticker : str
                ticker to be processed

            Returns
            -------
            pvt : DataFrame
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        # get close price
        hlc = self.ohlcv.loc[:, ticker].copy()

        return hlc["close"].diff(days).shift(-days)

    def get_all_technicals(self, ticker: str, returns_period: int = 7) -> DataFrame():
        '''
        Returns pd.DataFrame with all Technical Indicators

            Parameters
            ----------
            ticker : str
                ticker to be processed

            Returns
            -------
            pvt : DataFrame
        '''

        self.validate_ticker(ticker=ticker)  # validate ticker

        self.ohlcv[ticker, 'rsi'] = self.rsi(ticker)
        self.ohlcv[ticker, 'williams'] = self.williams(ticker)
        self.ohlcv[ticker, 'mfi'] = self.mfi(ticker)
        self.ohlcv[ticker, 'stoch_k'] = self.stoch(ticker)
        self.ohlcv[ticker, 'macd'] = self.macd(ticker)
        self.ohlcv[ticker, 'ma_10'] = self.ma(ticker, days=10)
        self.ohlcv[ticker, 'ma_50'] = self.ma(ticker, days=50)
        self.ohlcv[ticker, 'ma_200'] = self.ma(ticker, days=200)
        self.ohlcv[ticker, 'ema_7'] = self.ema(ticker, days=7)
        self.ohlcv[ticker, 'ema_14'] = self.ema(ticker, days=14)
        self.ohlcv[ticker, 'ema_21'] = self.ema(ticker, days=21)
        self.ohlcv[ticker, 'highlow'] = self.highlow(ticker)
        self.ohlcv[ticker, 'closeopen'] = self.closeopen(ticker)
        self.ohlcv[ticker, 'bb_high'], self.ohlcv[ticker, 'bb_low'] = self.bollingerbands(ticker)
        self.ohlcv[ticker, 'pvt'] = self.pvt(ticker)
        #         self.ohlcv[ticker, 'daily_return'] = self.daily_return(ticker)
        #         self.ohlcv[ticker, 'cum_daily_return'] = self.cum_daily_return(ticker)
        #         self.ohlcv[ticker, 'lagging_returns'] = self.lagging_returns(ticker, days=returns_period)

        return self.ohlcv[ticker]


def test():
    pass


def main():
    pass


if __name__ == '__main__':
    main()