### Required Libraries ###
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests


### Functionality Helper Functions ###
def parse_float(n):
    """
    Securely converts a non-numeric value to float.
    """
    try:
        return float(n)
    except ValueError:
        return float("nan")


def get_user(email):
    """
    Checks if the user exist or not in the api / database
    """
    response = requests.get(f"http://ec2-3-84-237-106.compute-1.amazonaws.com/stocksanalysis/check-user/{email}")
    response_json = response.json()
    return response_json['Body']['user-id']


def parse_user(user):
    """
    Parse the user data to the database / api
    """
    response = requests.get(f"http://ec2-3-84-237-106.compute-1.amazonaws.com/stocksanalysis/parse-user?email={user['email']}&name={user['name']}")
    response_json = response.json()
    return response_json

def parse_stocks(user):
    """
    Parse the tickers to the portfolio
    """
    response = requests.get(f"http://ec2-3-84-237-106.compute-1.amazonaws.com/stocksanalysis/parse-stocks?email={user['email']}&tickers={user['tickers']}")
    response_json = response.json()
    return response_json


def build_validation_result(is_valid, violated_slot, message_content):
    """
    Defines an internal validation message structured as a python dictionary.
    """
    if message_content is None:
        return {"isValid": is_valid, "violatedSlot": violated_slot}

    return {
        "isValid": is_valid,
        "violatedSlot": violated_slot,
        "message": {"contentType": "PlainText", "content": message_content},
    }


def validate_data(email,tickers, intent_request):
    """
    Validates the data provided by the user.
    """

    # Validate the email and name
    if email is None:
        return build_validation_result(
            False,
            "email",
            "The email cannot be empty, "
            "please provide an email",
        )

    # Validate the tickers
    if tickers is not None:
        tmp_list = tickers.split(';')
        if len(tmp_list) <= 0:
            return build_validation_result(
                False,
                "Tickers",
                "The tickers should be separate by semicolon(FB;MSFT; ... ) , "
                "Please specify properly the tickers.",
            )

    # A True results is returned if age or amount are valid
    return build_validation_result(True, None, None)


### Dialog Actions Helper Functions ###
def get_slots(intent_request):
    """
    Fetch all the slots and their values from the current intent.
    """
    return intent_request["currentIntent"]["slots"]


def elicit_slot(session_attributes, intent_name, slots, slot_to_elicit, message):
    """
    Defines an elicit slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "ElicitSlot",
            "intentName": intent_name,
            "slots": slots,
            "slotToElicit": slot_to_elicit,
            "message": message,
        },
    }


def delegate(session_attributes, slots):
    """
    Defines a delegate slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {"type": "Delegate", "slots": slots},
    }


def close(session_attributes, fulfillment_state, message):
    """
    Defines a close slot type response.
    """

    response = {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": fulfillment_state,
            "message": message,
        },
    }

    return response


### Intents Handlers ###
def stock_analysis(intent_request):
    name = get_slots(intent_request)["name"]
    email = get_slots(intent_request)["email"]
    tickers = get_slots(intent_request)["tickers"]

    # Gets the invocation source, for Lex dialogs "DialogCodeHook" is expected.
    source = intent_request["invocationSource"]  #

    if source == "DialogCodeHook":
        # This code performs basic validation on the supplied input slots.
        # Gets all the slots
        slots = get_slots(intent_request)

        # Validates user's input using the validate_data function
        validation_result = validate_data(email,tickers, intent_request)

        # If the data provided by the user is not valid,
        # the elicitSlot dialog action is used to re-prompt for the first violation detected.
        if not validation_result["isValid"]:
            slots[validation_result["violatedSlot"]] = None  # Cleans invalid slot

            # Returns an elicitSlot dialog to request new data for the invalid slot
            return elicit_slot(
                intent_request["sessionAttributes"],
                intent_request["currentIntent"]["name"],
                slots,
                validation_result["violatedSlot"],
                validation_result["message"],
            )

        # Fetch current session attributes
        output_session_attributes = intent_request["sessionAttributes"]

        # Once all slots are valid, a delegate dialog is returned to Lex to choose the next course of action.
        return delegate(output_session_attributes, get_slots(intent_request))

    # Create the user if doesn't exist
    user = {}
    user['name'] = name
    user['email'] = email
    user['tickers'] = tickers

    user_id = get_user(user['email'])

    if user_id is None:
        status = parse_user(user)
        if status['StatusCode'] == 200:
            parse_stocks(user)
            
    else:
        parse_stocks(user)

    # Return a message with conversion's result.
    return close(
        intent_request["sessionAttributes"],
        "Fulfilled",
        {
            "contentType": "PlainText",
            "content": "Your stocks will be analyze and you will get an update soon by email"
        },
    )


### Intents Dispatcher ###
def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    # Get the name of the current intent
    intent_name = intent_request["currentIntent"]["name"]

    # Dispatch to bot's intent handlers
    if intent_name == "StocksAnalyzer":
        return stock_analysis(intent_request)

    raise Exception("Intent with name " + intent_name + " not supported")


### Main Handler ###
def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """

    return dispatch(event)
