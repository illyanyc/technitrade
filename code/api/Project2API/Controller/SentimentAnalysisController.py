from Project2API.Extensions.ResponseExtension import Ok, InternalServerError
from flask_classful import route, FlaskView
from Project2Application.Service.SentimentAnalysisService import SentimentAnalysisService

class SentimentAnalysisController(FlaskView):
    def __init__(self):
        self.sentimentAnalysisService = SentimentAnalysisService()


    @route("/news-sentiment/<string:text>", methods=['GET'])
    def ReturnNewsSentiment(self, text: str):
        try:
            data = self.sentimentAnalysisService.ReturnNewsSentiment(text)
            return Ok(data)
        except ValueError:
            return InternalServerError(ValueError)

    @route("/twitter-sentiment/<string:ticker>", methods=['GET'])
    def SuporteResistencia(self, ticker: str):
        try:
            data = self.sentimentAnalysisService.ReturnTwitterSentiment(ticker)
            return Ok(data)
        except ValueError:
            return InternalServerError(ValueError)

