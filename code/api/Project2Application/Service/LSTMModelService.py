import datetime as dt
from os import getenv
from newsapi import NewsApiClient
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey,Date, update
import os
import pandas as pd
import Project2Application.Tools.technicals as technicals
import Project2Application.Tools.alpacas as alpaca
import Project2Application.Tools.lstm_model as lstm_model
from Project2Application.Tools.AWS_client import upload_to_aws,download_from_aws
from Project2Application.Service.SentimentAnalysisService import SentimentAnalysisService
from Project2Application.Tools.SendGridEmailService import SendEmail

class LTSMModelService:

    def __init__(self):
        # self.engine = create_engine("postgresql://user_project2:2021Project2@project2.cgipq7lut6ku.us-east-1.rds.amazonaws.com:5432/project2")
        self.engine = create_engine(
            "postgresql://xxxxxxxxxx")

        self.newsApi = NewsApiClient(api_key=getenv('NEWSAPIKEY'))


    def CheckModelExist(self,ticker):
        query = f"select * from models where ticker = '{ticker}'"
        data = pd.read_sql(query, self.engine)
        if len(data) > 0:
            return True
        else:
            return False

    def ParseModelCreatedDB(self,ticker,path):
        metadata = MetaData()
        table = Table('models', metadata,
                      Column('ticker', String(100)),
                      Column('path', String(200)),
                      Column('creation_date', Date),
                      Column('news_sentiment', String(100)),
                      Column('twitter_sentiment', String(100)),
                      Column('recommendation', String(100)),

                      )
        result = self.engine.execute(table.insert().returning(Column('ticker')), {'ticker': ticker, 'path': path,'creation_date' : dt.datetime.now(),
                                                                                  'news_sentiment' : '','twitter_sentiment' : '' , 'recommendation' : ''})


    def ReturnPedingPortfolioDb(self):
        query = f"SELECT A.USER_ID,A.NAME,A.EMAIL,B.PORTFOLIO_ID,B.TICKER FROM USERS AS A INNER JOIN USER_PORTFOLIO AS B ON A.USER_ID = B.USER_ID WHERE B.STATUS = 0"
        data = pd.read_sql(query, self.engine)
        if len(data) > 0:
            return data
        else:
            return None

    def ReturnModels(self):
        query = f"SELECT * FROM MODELS WHERE RECOMMENDATION = ''"
        data = pd.read_sql(query, self.engine)
        if len(data) > 0:
            return data
        else:
            return None
    def UpdateModelsRecommendation(self,ticker,news_sentiment,twitter_sentiment,recommendation):
        meta = MetaData()
        model = Table(
            'models',
            meta,
            Column('ticker', Integer, primary_key=True),
            Column('news_sentiment', String),
            Column('twitter_sentiment', String),
            Column('recommendation', String),
        )
        conn = self.engine.connect()
        stmt = model.update().where(model.c.ticker == ticker).values(news_sentiment=news_sentiment,twitter_sentiment = twitter_sentiment,recommendation = recommendation)
        conn.execute(stmt)

    def UpdateStatusPortfolio(self,portfolio_id):
        meta = MetaData()
        user_portfolio = Table(
            'user_portfolio',
            meta,
            Column('portfolio_id', Integer, primary_key=True),
            Column('status', Integer),
        )
        conn = self.engine.connect()
        stmt = user_portfolio.update().where(user_portfolio.c.portfolio_id == portfolio_id).values(status=1)
        conn.execute(stmt)

    def ExtractTickers(self):
        portfolio = self.ReturnPedingPortfolioDb()
        tickers_pending = []
        if portfolio is not None:
            for x in range(0, len(portfolio)):
                tmp_tickers = portfolio.iloc[x, 4].split(';')
                if len(tmp_tickers) > 0:
                    for ticker in tmp_tickers:
                        tickers_pending.append(str(ticker).upper())
        if len(ticker) > 0:
            return set(tickers_pending)

    # LSTM Helper Functions

    def ForecastModel(self,ticker):
        ## Forecasting Future Stock Price
        from datetime import date, datetime, timedelta

        pred_end_date = datetime.now()
        pred_start_date = (pred_end_date - timedelta(days=200))

        pred_start_date = pred_start_date.strftime('%Y-%m-%d')
        pred_end_date = pred_end_date.strftime('%Y-%m-%d')

        print(f"Forecast start date : {pred_start_date}")
        print(f"Forecast end date : {pred_end_date}")

        # Load the dataset
        pred_ohlcv_df = alpaca.ohlcv(ticker, start_date=pred_start_date, end_date=pred_end_date)
        pred_tech_ind = technicals.TechnicalAnalysis(pred_ohlcv_df)

        pred_df = pred_tech_ind.get_all_technicals(ticker)
        forecast_model = lstm_model.ForecastPrice(pred_df)

        if self.CheckModelExist(ticker) == True:
            #local_path_model = f'C:/Repos/model/{ticker}_model.h5'
            local_path_model = f"/tmp/{ticker}_model_model.h5"
            download_from_aws(f"{ticker}_model.h5", local_path_model)
            forecast_model.load_model(local_path_model)
            preds = forecast_model.forecast()
            if len(preds) > 0:
                os.remove(local_path_model)
                return preds
        else:
            return None



    def ReturnRecommendations(self):
        query = f"SELECT * FROM MODELS WHERE RECOMMENDATION <> ''"
        data = pd.read_sql(query, self.engine)

        if len(data) > 0:
            dict_tickers = {}
            for x in range(0,len(data)):
                dict_tickers[data.iloc[x,0]] = {'news_sentiment' : data.iloc[x,3],'twitter_sentiment': data.iloc[x,4], 'advice' : data.iloc[x,5]}
            return dict_tickers
        else:
            return None


    def AnalyzeTickersPredictions(self):
        users_portfolio = self.ReturnPedingPortfolioDb()


    def AnalyzeStocksRecommendation(self):
        models = self.ReturnModels()
        if models is not None:
            for x in range(0, len(models)):
                ticker = models.iloc[x, 0]

                preds = self.ForecastModel(ticker)
                if len(preds) > 0:
                    current_price = preds.iloc[0, 0]
                    last_price = preds.iloc[-1, 0]

                    if last_price > current_price:
                        recommendation = 'BUY'
                    else:
                        recommendation = 'SELL'

                    # Sentiment Analysis
                    news_sentiment = SentimentAnalysisService.ReturnNewsSentiment(self, ticker)['sentiment']
                    twitter_sentiment = SentimentAnalysisService.ReturnTwitterSentiment(self, ticker)['sentiment']

                    # Updating the table
                    self.UpdateModelsRecommendation(ticker, news_sentiment, twitter_sentiment, recommendation)
                    print(ticker)


    def SendEmailUsers(self):
        user = {}
        user['email'] = 'fdobastos@gmail.com'
        SendEmail(user)



    def UpdateUserPortfolio(self):
        user_portfolios = self.ReturnPedingPortfolioDb()
        stocks_recommendation = self.ReturnRecommendations()

        for i,k in user_portfolios.iterrows():
            tickers = str(k['ticker']).split(';')
            name = k['name']
            email = str(k['email']).lower()
            list_user_portfolio_result = []
            qty_tickers_processed = 0
            if tickers is not None:
                for ticker in tickers:
                    if str(ticker).upper() in stocks_recommendation:
                        list_user_portfolio_result.append({'ticker' : str(ticker).upper(), 'results' : stocks_recommendation[str(ticker).upper()]})
                    else:
                        break
            else:
                continue

            if len(list_user_portfolio_result) == len(tickers):
                SendEmail(name,email,list_user_portfolio_result)
                self.UpdateStatusPortfolio(k['portfolio_id'])



    def TriggerModel(self):
        tickers = self.ExtractTickers()
        from datetime import date, datetime, timedelta
        end_date = datetime.now()
        start_date = (end_date - timedelta(days=1000))

        print(f"Start date : {start_date}")
        print(f"End date : {end_date}")

        for ticker in tickers:
            if self.CheckModelExist(ticker) == False:
                ohlcv_df = alpaca.ohlcv(ticker, start_date=start_date, end_date=end_date)
                tech_ind = technicals.TechnicalAnalysis(ohlcv_df)
                df = tech_ind.get_all_technicals(ticker)
                model = lstm_model.MachineLearningModel(df)
                hist = model.build_model(summary=0, verbose=0)

                #path_model = f"C:/Repos/model/{ticker}_model"
                path_model = f"/tmp/{ticker}_model"
                model.save_model(path_model)
                aws_file = upload_to_aws(f"{path_model}.h5")

                if aws_file is not None:
                    self.ParseModelCreatedDB(ticker, str(aws_file))
                    os.remove(f"{path_model}.h5")
            else:
                continue

        self.AnalyzeStocksRecommendation()
        self.UpdateUserPortfolio()

        return {'status' : 'completed'}
















































