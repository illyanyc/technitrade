from os import getenv
from pymongo import MongoClient


class NewsAPI:
    def __init__(self):
        self.apikey = getenv('NEWSAPIKEY')
