import datetime as dt
from os import getenv
from newsapi import NewsApiClient
import datetime as dt
import re
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
import pandas as pd


class StocksAnalysisService:

    def __init__(self):
        self.newsApi = NewsApiClient(api_key=getenv('NEWSAPIKEY'))
        #self.engine = create_engine("postgresql://user_project2:2021Project2@project2.cgipq7lut6ku.us-east-1.rds.amazonaws.com:5432/project2")
        self.engine = create_engine("postgresql://postgres:mypostgrespassword@project3.ccetvexupnnw.us-east-1.rds.amazonaws.com:5432/project2")


    def CheckIfUserExist(self,email):
        query = f"select * from users where email = '{email}'"
        data = pd.read_sql(query,self.engine)
        if len(data) > 0:
            return {'user-id' : str(data.iloc[0,0])}
        else:
            return{'user-id' : None}

    def ParseUserToDb(self,user):
        metadata = MetaData()
        table = Table('users', metadata,
                      Column('name', String(100)),
                      Column('email', String(200)),
                      )

        result = self.engine.execute(table.insert().returning(Column('user_id')), {'name' : user['name'],'email': user['email']})
        for id in result:
            user_id = int(id[0])
        return {'creation' : 'Ok','user-id' : user_id}

    def ParseUserStocks(self,user):
        email = user['email']
        tickers = user['tickers']
        user = pd.read_sql(f"select * from users where email = '{email}'",self.engine)

        if len(user) >0:
            user_id = int(user.iloc[0,0])
        else:
            metadata = MetaData()
            table = Table('users', metadata,
                          Column('name', String(100)),
                          Column('email', String(200)),
                          )
            result = self.engine.execute(table.insert().returning(Column('user_id')), {'name': 'GUEST', 'email': email})
            for id in result:
                user_id = int(id[0])

        metadata = MetaData()
        table = Table('user_portfolio', metadata,
                      Column('user_id', Integer()),
                      Column('ticker', String(500)),
                      Column('status', Integer()),
                      )
        self.engine.execute(table.insert(), {'user_id': user_id, 'ticker':str(tickers).upper(),'status' : 0})
        return {'creation': 'Ok'}
