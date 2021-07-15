from Project2API.Extensions.ResponseExtension import Ok, InternalServerError
from flask_classful import route, FlaskView
from Project2Application.Service.StocksAnalysisService import StocksAnalysisService
from flask import request


class StocksAnalysisController(FlaskView):
    def __init__(self):
        self.stocksAnalysisService = StocksAnalysisService()

    @route("/check-user/<string:email>", methods=['GET'])
    def CheckIfUserExist(self, email: str):
        try:
            data = self.stocksAnalysisService.CheckIfUserExist(email)
            return Ok(data)
        except ValueError:
            return InternalServerError(ValueError)

    @route("/parse-user", methods=['GET'])
    def ParseUserToDb(self):
        try:
            name = request.args.get('name', None)
            email = request.args.get('email', None)

            if len(name) >= 0 or len(email) >=0:
                user = {}
                user['name'] = name
                user['email'] = email
                data = self.stocksAnalysisService.ParseUserToDb(user)
                return Ok(data)
            else:
                {'error' : 'Parameters invalid'}

        except ValueError:
            return InternalServerError(ValueError)


    @route("/parse-stocks", methods=['GET'])
    def ParseUserStocks(self):
        try:
            email = request.args.get('email', None)
            tickers = request.args.get('tickers', None)

            if email is not None and tickers is not None:
                if len(email) >= 0 and len(tickers) >= 0:
                    user = {}
                    user['email'] = email
                    user['tickers'] = tickers
                    data = self.stocksAnalysisService.ParseUserStocks(user)
                    return Ok(data)
                else:
                    return {'error' : 'invalid parameter'}
            else:
                return {'error' : 'invalid parameter'}



        except ValueError:
            return InternalServerError(ValueError)