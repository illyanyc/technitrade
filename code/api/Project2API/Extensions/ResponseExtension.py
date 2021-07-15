from flask import jsonify


def Ok(objectResponse):
    return {
        'StatusCode': 200,
        'Message': 'OK',
        'Body': objectResponse
    }, 200

def Created(objectResponse):
    return {
        'StatusCode': 201,
        'Message': 'Created',
        'Body': objectResponse
    }, 201

def InternalServerError(objectResponse):
    return {
        'StatusCode': 500,
        'Message': 'InternalServerError',
        'Body': jsonify(objectResponse)
    }, 500
