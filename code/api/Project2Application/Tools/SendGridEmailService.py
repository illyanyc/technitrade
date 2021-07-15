import base64
import sendgrid
from sendgrid.helpers.mail import *
# SendGrid
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

SENDGRID_API_KEY = 'xxxxxxxxxxxxxx'


def SendEmail(name,email,list_user_portfolio_result):

    text = f"Hi {name}, " + '\n'
    text += '<br></br>'
    text += '<br></br>'
    text +='Your portfolio analysis: ' + '\n'
    text += '<br></br>'
    text += '<br></br>'
    text += '<table border = "1"><tr><th>Ticker</th><th>News Sentiment</th><th>Twitter Sentiment</th><th>AI Opinion</th></tr>'

    for x in list_user_portfolio_result:

        if x['results']['advice'] == 'SELL':
            text += f"<tr align='center'><td><a href='https://finance.yahoo.com/quote/{x['ticker']}'>{x['ticker']}</a></td><td>{x['results']['news_sentiment']}</td><td>{x['results']['twitter_sentiment']}</td><td bgcolor='#fac8c8'>{x['results']['advice']}</td></tr>"
        else:
            text += f"<tr align='center'><td><a href='https://finance.yahoo.com/quote/{x['ticker']}'>{x['ticker']}</a></td><td>{x['results']['news_sentiment']}</td><td>{x['results']['twitter_sentiment']}</td><td bgcolor='#c8fac8'>{x['results']['advice']}</td></tr>"

    text += '</table>'
    text += '<br></br>'
    text += '<br></br>'
    text += '<br></br>'
    text += '<strong>All the analysis and recommendations are done by AI (Artificial Intelligence) </strong>'

    message = Mail(
        from_email= 'technitradeservice@gmail.com',
        to_emails=email,
        subject='Technitrade - Stock Analysis',

        html_content=text)
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
    except Exception as e:
        print(e.message)

