import pandas as pd
import re
import datetime as dt
import json
import requests
import json
import os
from google.oauth2.credentials import Credentials
from pathlib import Path
import tweepy

consumer_key = 'xxxxxxxxxx'
consumer_secret = 'xxxxxxxxxxxxx'
access_token = 'xxxxxxxxx'
access_token_secret = 'xxxxxxxxxx'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub('[\W_]+',' ',text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = text.replace('https','')
    text = text.replace('  t co ','')
    return text


from google.cloud import language_v1

def GetSentimentAnalysisGoogle(text_content):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str('Project2Application/Tools/service-account.json')
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    document = {'content': text_content, 'type_': type_}
    encoding_type = language_v1.EncodingType.UTF8
    response = client.analyze_sentiment(request={'document': document, 'encoding_type': encoding_type})
    return response.document_sentiment.score


def ReturnNewsSentiment(self,ticker):
    list_score_news = []
    begin_date = dt.datetime.strftime((dt.datetime.now() - dt.timedelta(days=30)),'%Y-%m-%d')
    end_date = dt.datetime.strftime(dt.datetime.now(),'%Y-%m-%d')
    all_articles =  self.newsApi.get_everything(q=ticker,
                                              from_param=begin_date,
                                              to=end_date,
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=12)
    for x in range(0,len(all_articles['articles'])):
        news = all_articles['articles'][x]
        list_score_news.append(GetSentimentAnalysisGoogle(news['description']))
        final_score = round(sum(list_score_news)/len(list_score_news),2)

        if len(list_score_news) > 0:
            if final_score >= 0.25:
                sentiment = 'POSITIVE'
            elif final_score >= 0.20 and final_score < 0.25:
                sentiment = 'NEUTRAL-POSITIVE'
            elif final_score < 0.20 and final_score >= -0.25:
                sentiment = 'NEUTRAL'
            elif final_score < -0.25:
                sentiment = 'NEGATIVE'
    return {'score' : final_score, 'sentiment' :sentiment}


def ReturnTwitterSentiment(self,ticker):
    hashtag = f"${ticker}  -filter:retweets"
    tweets = tweepy.Cursor(api.search, q=hashtag).items(50)
    list_score_twitter = []

    text_final = ''
    for tweet in tweets:
        if tweet.lang == "en":
            text = cleanTxt(tweet.text).lower()
            text_final += text
    final_score = GetSentimentAnalysisGoogle(text_final)
    if final_score >= 0.25:
        sentiment = 'POSITIVE'
    elif final_score >= 0.20 and final_score < 0.25:
        sentiment = 'NEUTRAL-POSITIVE'
    elif final_score < 0.20 and final_score >= -0.25:
        sentiment = 'NEUTRAL'
    elif final_score < -0.25:
        sentiment = 'NEGATIVE'

    return {'sentiment': sentiment, 'score': round(final_score,2)}







