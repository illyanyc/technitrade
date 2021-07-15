from Project2API.Extensions.ResponseExtension import Ok, InternalServerError
from flask_classful import route, FlaskView
from Project2Application.Service.LSTMModelService import LTSMModelService
from flask import request


class LSTMModelController(FlaskView):
    def __init__(self):
        self.lstmModelService = LTSMModelService()

    @route("/trigger-model", methods=['GET'])
    def TriggerModel(self):
        try:
            data = self.lstmModelService.TriggerModel()
            return Ok(data)
        except ValueError:
            return InternalServerError(ValueError)


    @route("/results-model", methods=['GET'])
    def ReturnRecommendations(self):
        try:
            data = self.lstmModelService.ReturnRecommendations()
            return Ok(data)
        except ValueError:
            return InternalServerError(ValueError)






