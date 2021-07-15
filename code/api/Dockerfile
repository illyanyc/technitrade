FROM python:3.8.3-slim-buster

WORKDIR /src
EXPOSE 80

RUN pip install --upgrade pip
RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENV FLASK_APP=app.py FLASK_ENV=Production DEBUG=False HOST="0.0.0.0" APP_PORT=80

COPY . .

ADD . .

CMD ["python","./app.py"]