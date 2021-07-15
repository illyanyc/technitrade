from os import getenv
from dotenv import load_dotenv


class DevelopmentConfig:

    load_dotenv()
    FLASK_ENV = getenv('FLASK_ENV')
    DEBUG = True
    HOST = getenv('HOST')
    APP_PORT = int(getenv('APP_PORT'))
    FLASK_APP = getenv('FLASK_APP')
    JWT_SECRET = getenv('JWT_SECRET')


class Production:
    FLASK_ENV = getenv('FLASK_ENV')
    DEBUG = False
    HOST = getenv('HOST')
    APP_PORT = int(getenv('APP_PORT'))
    FLASK_APP = getenv('FLASK_APP')
    JWT_SECRET = getenv('JWT_SECRET')


config = {
    'Development': DevelopmentConfig,
    'Production': Production,
    'Default': DevelopmentConfig
}
