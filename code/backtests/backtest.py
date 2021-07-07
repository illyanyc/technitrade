from datetime import datetime
import numpy as np
import pandas as pd
import backtrader as bt
import yahoo as yh

import pytz
IST = pytz.timezone('America/New_York')
timeformat = "%m/%d/%Y %H:%M:%S"
dateformat = "%Y-%m-%d"

from pos_sizers import LongOnly

def optimization(strat,
                 ticker : str,
                 start_cash : float,
                 broker_comm : float,
                 risk : float,
                 period : list, 
                 upperband : list,
                 lowerband : list,
                 start_date,
                 end_date):
    
    start_time = datetime.now(IST)
    
    ohlcv = yh.ohlcv(ticker, start_date, end_date)

    # parse ohlcv data into PandasData 
    data = bt.feeds.PandasData(dataname=ohlcv)

    # instantiate Cerebro
    cerebro = bt.Cerebro(optreturn=False)
    
    # add strategy
    cerebro.optstrategy(strat,
                        period=period,
                        upperband=upperband,
                        lowerband=lowerband)
        
    # pass data to cerebro
    cerebro.adddata(data)

    # set starting account value and broker commission
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=broker_comm)

    # add a FixedSize sizer according to the stake
    cerebro.addsizer(LongOnly, risk=risk)

    # start backtest
    opt_run = cerebro.run()

    # calculate hold open
    hold_size = int((start_cash * risk) / ohlcv.iloc[0]['Open'])
    hold_open = (hold_size * ohlcv.iloc[0]['Open']) - (hold_size * broker_comm)
    # calculate hold close
    hold_close = (hold_size * ohlcv.iloc[-1]['Close']) - (hold_size * broker_comm)
    # calculate hold pnl
    hold_pnl = round((hold_close - hold_open),2)

    # unpack backtest data
    final_results_list = []
    for run in opt_run:
        for strategy in run:
            value = round(strategy.broker.get_value(),2)
            trade_pnl = round(value - start_cash,2)
            alpha = round((trade_pnl - hold_pnl),2)
            pct_alpha = round((alpha/hold_pnl)*100,2)

            period_instance = strategy.params.period
            upperband_instance = strategy.params.upperband
            lowerband_instance = strategy.params.lowerband

            results = [period_instance, upperband_instance, lowerband_instance,trade_pnl,hold_pnl,alpha,pct_alpha]
            final_results_list.append(results)

    result_df = pd.DataFrame(final_results_list, 
              columns=['period',
                       'upperband',
                       'lowerband',
                       'trade_pnl',
                       'hold_pnl',
                       'alpha',
                       'pct_alpha'])
    
    end_time = datetime.now(IST)
    delta = end_time - start_time
    print(f"Runtime : {delta}")

    return result_df