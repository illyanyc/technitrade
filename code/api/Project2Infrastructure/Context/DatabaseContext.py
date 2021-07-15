from os import getenv
from pymongo import MongoClient


class MongoDbContext:
    def __init__(self):
        self.connectionString = getenv('CONNECTIONSTRING')
        self.mongoClient = MongoClient(self.connectionString)
        self.mongoDatabase = self.mongoClient[getenv('DATABASE_NAME')]
