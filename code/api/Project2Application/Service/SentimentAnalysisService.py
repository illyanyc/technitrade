import datetime as dt
from os import getenv
from newsapi import NewsApiClient
import datetime as dt
import re
from google.cloud import language_v1
from google.oauth2.credentials import Credentials
import Project2Application.Tools.SentimentAnalysisTools as tools

class SentimentAnalysisService:

    def __init__(self):
        self.newsApi = NewsApiClient(api_key=getenv('NEWSAPIKEY'))



    def ReturnNewsSentiment(self,text):
        if len(text) > 0:
            score = tools.ReturnNewsSentiment(self,text)
            return score
        else:
            return {'error' : 'Text should not be empty'}


    def ReturnTwitterSentiment(self,ticker):
        if len(ticker) > 0:
            score = tools.ReturnTwitterSentiment(self,ticker)
            return score
        else:
            return {'error': 'Text should not be empty'}

    def ReturnFullDataSentiment(self):
        pass

