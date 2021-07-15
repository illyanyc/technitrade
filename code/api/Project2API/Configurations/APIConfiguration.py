from flask import Flask
from Project2API.Configurations.EnvConfiguration import config


def CreateApp(environment: str) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config[environment])

    return app
