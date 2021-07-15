![header](img/logo.svg)


---
**Disclosure**

:warning:   NOT INVESTMENT ADVICE   :warning:

The content produced by this application is for informational purposes only, you should not construe any such information or other material as legal, tax, investment, financial, or other advice. Nothing contained in this article, Git Repo or withing the output produced by this application constitutes a solicitation, recommendation, endorsement, or offer by any member involved working on this project, any company they represent or any third party service provider to buy or sell any securities or other financial instruments in this or in in any other jurisdiction in which such solicitation or offer would be unlawful under the securities laws of such jurisdiction. 

The use of word "recommendation", "opinion" in this article or any other word with a similar meaning, within the application, or within information produced by the application is for demonstration purposes only, and is not a recommendation to buy or sell any securities or other financial instruments!

This application was created solely to satisfy the requirements of Columbia University FinTech Bootcamp Project #2 Homework, and the results produced by this application may be incorrect.

---

### Table of Contents
* [Overview](#overview)
* [Application Logic](#application-logic)
* [Libraries](#libraries)
* [Flask API](#flask-api)
* [SQL Database](#sql-database)
* [Interface](#interface)
* [Technical Analysis](#technical-analysis)
* [Machine Learning Model](#machine-learning-model)
* [Sentiment Analysis](#sentiment-analysis)
* [Team](#team)

---
# Overview

Technitrade lets user track a portfolio of stocks, periodically getting News Sentiment, Twitter Sentiment, and Machine Learning AI Stock Opinion. The machine learning model calculates "opinion" based on market data and technical analysis, while the investor sentiment calculated by natural language processing analysis of recent news articles and Tweets.

The user interacts with the program via an [Amazon Lex chatbot](#aws-interface). The machine learning analysis is performed using [LSTM (Long Short-Term Memory) model](#machine-learning). The model is trained on [technical analysis indicators](#technical-analysis). Sentiment analysis is performed by [Google Cloud Natural Language](#sentiment-analysis) using NewsAPI and Twitter APIs as data source.


**Demo Jupyter Notebooks**
1. Technical Analysis Demo : <code>[technicals_demo.ipynb](code/technicals/technicals_demo.ipynb)</code>
2. Machine Learning Demo : <code>[lstm_demo.ipynb](code/ml/lstm_demo.ipynb)</code>
3. Sentiment Analysis Demo : <code>[nlp_demo.ipynb](code/nlp/nlp_demo.ipynb)</code>

**Production Code**
Production API, Application and Infrastructure code, along with a Docker container can be found here: [<code>code/api/</code>](https://github.com/illyanyc/technitrade/tree/main/code/api)

---

# Application Logic

![flowchart](img/flowchart.png)

---

# Libraries

The following libraries are used:

### Data Computation and Visualization
* [Numpy](https://numpy.org/) - "The fundamental package for scientific computing with Python".
* [Pandas](https://pandas.pydata.org/) - data analysis and manipulation tool.

```python
pip install pandas
```

* [Matplotlib](https://matplotlib.org/) - Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

```python
pip install matplotlib
```

### Database
* [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) - AWS SDK for Python to create, configure, and manage AWS services, such as Amazon Elastic Compute Cloud (Amazon EC2) and Amazon Simple Storage Service (Amazon S3). The SDK provides an object-oriented API as well as low-level access to AWS services.

```python
pip install boto3
```

* [psycopg2](https://www.psycopg.org/docs/) - PostgreSQL database adapter for the Python programming language.

```python
pip install psycopg2
```

### Data Source APIs
* [Dotenv](https://pypi.org/project/python-dotenv/) - ython-dotenv reads key-value pairs from a .env file and can set them as environment variables. 

```python
pip install python-dotenv
```

* [Alpaca Trade API](https://alpaca.markets/docs/) - Internet brokerage and market data connection service.

```python
pip install alpaca-trade-api
```

* [NewsAPI](https://newsapi.org/) - NewsAPI locates articles and breaking news headlines from news sources and blogs across the web and returns them as JSON.

```python
pip install newsapi-python
```

* [Twitter API](https://developer.twitter.com/en/docs) - Twitter API enables programmatic access to Twitter.
    * [tweepy](https://www.tweepy.org/) - An easy-to-use Python library for accessing the Twitter API.
    
```python
pip install tweepy
```

### Machine Learning
* [Scikit-Learn](https://scikit-learn.org/stable/) - Machine learning library for python

```python
pip install scikit-learn
```

* [Tensorflow](https://www.tensorflow.org/) - an end-to-end open source platform for machine learning.

```python
pip install tensorflow
```

* [Keras](https://keras.io/) - Python API used to interact with Tensorflow.

```python
pip install keras
```

* [NLTK](https://www.nltk.org/) - NLTK is a leading platform for building Python programs to work with human language data.

```python
pip install nltk
```

* [Google Cloud language_v1](https://cloud.google.com/natural-language/docs/apis) - an API that connects to Google Cloud Natural Language

```python
pip install google-cloud-language
```

### Technical Analysis Library
* technitrade - a custom built library for technical analysis.

### Other Development Frameworks
* [Flask](https://flask.palletsprojects.com/en/2.0.x/) - Flask is a micro web framework written in Python.
* [AWS Lex Bot](https://aws.amazon.com/lex/) - Amazon Lex is a service for building conversational interfaces into any application using voice and text. 
* [Twilio SendGrid](https://sendgrid.com/) - communication platform for transactional and marketing email.
---

# Interface

User interfaces with the application using Amazon Lex Bot.
Amazon Lex Bot gathers the following user info:

1. Name
2. Email
3. *n* number of portfolio stock tickers

The user gets the News Sentiment, Twitter Sentiment, and Machine Learning AI Stock Opinion via periodic emails. The first email is received right after the Machine Learning model finished training and is fitted with data to predict future stock prices.

The emails are distributed via Twilio's SendGrid service.

The resulting email looks something like this:

![result_email](img/result.png) 


---
# Flask API

<img src="img/flask.png" width=200>

## Overview
A Flask API was built in order to handle all tasks between the:

1. Amazon Lex Bot
2. Data sources: Market Data Connection (see <code>[code/marketdata/]</code> folder), NewsAPI, Twitter API
3. Technical Analysis module : [<code>technicals.py</code>](code/technicals/technicals.py)
4. Machine Learning module : [<code>lstm_model.py</code>](code/ml/lstm_model.py)
5. Sentiment Analysis service
6. Amazon RDS PostgreSQL server

All events are triggered by AWS Cloudwatch. AWS Lambda function handle all of the production python code.

* Flask API services can he found here: [<code>Project2API</code>](https://github.com/illyanyc/technitrade/tree/main/code/api/Project2API)

* Project Application code can be found here: [<code>Project2Application</code>](https://github.com/illyanyc/technitrade/tree/main/code/api/Project2Application)

* Project Infrastructure code can be found here: [<code>Project2Infrastructure</code>](https://github.com/illyanyc/technitrade/tree/main/code/api/Project2Infrastructure)


## Flask API steps

The steps by which the Flask API executes application workflow is outlines in the table below.

|   | Objective        | Action                                | Trigger                     |
|---|------------------|---------------------------------------|-----------------------------|
| 1 | User Data        | User & Portfolio Creation             | Amazon LEX                  |
| 2 | Model - Training | Trigger the API to run the training   | Lambda / CloudWatch         |
| 3 | Model - Training | Save the model in Amazon S3           | API                         |
| 4 | Model - Forecast | Forecast the tickers                  | Lambda / CloudWatch /   API |
| 5 | User Data        | Update the user portfolio             | Lambda / CloudWatch /   API |
| 6 | User Data        | Send email to the users               | Lambda / CloudWatch /   API |

---
# SQL Database

<img src="img/postgresql_logo.svg" width=300> 

## Database Overview
A [PostgreSQL](https://www.postgresql.org/) database hosted on [Amazon RDS](https://aws.amazon.com/rds/) is utilized to store all the user data and machine learning models. 

All database code can be viewed here: [<code>code/src/</code>](https://github.com/illyanyc/technitrade/tree/main/code/src)

### Amazon RDS
Amazon Relational Database Service (Amazon RDS) makes it easy to set up, operate, and scale a relational database in the cloud. It provides cost-efficient and resizable capacity while automating time-consuming administration tasks such as hardware provisioning, database setup, patching and backups. 

### Postgres
PostgreSQL is a powerful, open source object-relational database system with over 30 years of active development that has earned it a strong reputation for reliability, feature robustness, and performance.

[psycopg2](https://www.psycopg.org/docs/) was used to interface python with PostgreSQL database.
[pgAdmin](https://www.pgadmin.org/) was used for testing and debugging. 

## Database Schematics

![database_flowchart](img/database_flowchart.png)

---

# Technical Analysis

Technical analysis is performed via <code>technicals</code> module. A demonstration of the module can be seen in <code>[technicals_demo.ipynb](code/technicals/technicals_demo.ipynb)</code>

## Indicators

### Relative Strength Index (RSI)

RSI is a momentum indicator which measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock. [[Investopedia](https://www.investopedia.com/terms/r/rsi.asp)]

<details>
<summary> RSI Equation
</summary>
<br>
    <img src="img/equation_rsi.svg">
<br>    
where:<br>
    relative strenght (<i>RS</i>) = <i>average gain</i> - <i>average loss</i>
</details>

### William's Percent Range (Williams %R)

Williams %R is a momentum indicator which measures overbought and oversold levels. It has a domain between 0 and -100.The Williams %R may be used to find entry and exit points in the market. [[Investopedia](https://www.investopedia.com/terms/w/williamsr.asp)]

<details>
<summary> Williams %R Equation
</summary>
<br>
    <img src="img/equation_williams.svg">
<br>    
where:<br>
<i>Highest High</i> = Highest price in the lookback period.<br>
<i>Close</i> = Most recent closing price.<br>
<i>Lowest Low</i> = Lowest price in the lookback period.<br>
</details>
    

### Money Flow Index

The money flow index (MFI) is an oscillator that ranges from 0 to 100. It is used to show the money flow (an approximation of the dollar value of a day's trading) over several days. [[Wikipedia](https://en.wikipedia.org/wiki/Money_flow_index)]


<details>
<summary> Money Flow Index Equation
</summary>  
- Positive money flow is calculated by adding the money flow of all the days where the typical price is higher than the previous day's typical price.<br>
- Negative money flow is calculated by adding the money flow of all the days where the typical price is lower than the previous day's typical price.<br>
- If typical price is unchanged then that day is discarded.<br>
- The money flow is divided into positive and negative money flow.<br>
<br>
    <img src="img/equation_mfi_1.svg">
<br> 
    <img src="img/equation_mfi_2.svg">
<br> 
    <img src="img/equation_mfi_3.svg">
<br> 
    <img src="img/equation_mfi_4.svg">
<br> 
</details>


### Stochastic Oscillator

The stochastic oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time. The sensitivity of the oscillator to market movements is reducible by adjusting that time period or by taking a moving average of the result. It is used to generate overbought and oversold trading signals, utilizing a 0–100 bounded range of values. [[Investopedia](https://www.investopedia.com/terms/s/stochasticoscillator.asp)]

<details>
<summary> Stochastic Oscillator Equation
</summary>  
<br>
    <img src="img/equation_stoch.svg">
<br>
where:<br>
    <i>C</i> = The most recent closing price<br>
    <i>Low<sub>n</sub></i> = The lowest price traded of the <i>n</i> previous trading sessions<br>
    <i>High<sub>n</sub></i> = The highest price traded during the same <i>n</i>-day period<br>
    <i>%K</i> = The current value of the stochastic indicator<br>
</details>


### Moving Average Convergence Divergence (MACD)

MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period [exponential moving average (EMA)](#exponential-moving-average) from the 12-period EMA. [[Investopedia](https://www.investopedia.com/terms/m/macd.asp)]

<details>
<summary> MACD Equation
</summary>  
<br>
    <img src="img/equation_macd.svg">
<br>
    <a href="https://www.investopedia.com/terms/e/ema.asp">Exponential moving average</a> is a <a href="https://www.investopedia.com/terms/m/movingaverage.asp">moving average</a> that places a greater weight to most recent data points and less to the older data points. In finance, EMA reacts more significantly to recent price changes than a simple moving average (SMA)which applies an equal weight to all observations in the period.
In statistics, a moving average (MA), also known as simple moving average (SMA) in finance, is a calculation used to analyze data points by creating a series of averages of different subsets of the full data set. 
</details>


### Moving Average

The moving average is a calculation used to smooth data and in finance used as a stock indicator. [[Investopedia](https://www.investopedia.com/terms/m/movingaverage.asp)]

<details>
<summary> Moving Average Equation
</summary>  
<br>
    <img src="img/equation_ma.svg">
<br>
where:<br>
    <i>A</i> = Average in period <i>n</i><br>
    <i>n</i> = Number of time periods<br>
</details>


### Exponential Moving Average

The exponential moving average is a type of moving average that gives more weight to recent prices in an attempt to make it more responsive to new information. [[Investopedia](https://www.investopedia.com/terms/m/movingaverage.asp)]

<details>
<summary> EMA Equation
</summary>  
<br>
    <img src="img/equation_ema.svg">
<br>
where:<br>
    <i>EMA<sub>t</sub></i> = EMA today<br>
    <i>EMA<sub>y</sub></i> = = EMA yesterday<br>
    <i>V<sub>t</sub></i> = Value today<br>
    <i>s</i> = smoothing<br>
    <i>d</i> = number of days<br>
</details>


### High Low and Close Open

the high-low and close-open indicators are the difference between the high and low prices of the day and close and open prices of the day respectively.

<details>
<summary> High-Low and Close-Open Equations
</summary>  
<br>
<img src="img/equation_hl.svg"><br>
<img src="img/equation_oc.svg"><br>
</details>


### Bollinger Bands

A Bollinger Band® is a technical analysis tool defined by a set of trendlines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of a security's price. Bollinger Bands® were developed and copyrighted by famous technical trader John Bollinger, designed to discover opportunities that give investors a higher probability of properly identifying when an asset is oversold or overbought. [[Bollinger Bands](https://www.bollingerbands.com/bollinger-bands)],[[Investopedia](https://www.investopedia.com/terms/b/bollingerbands.asp)]

<details>
<summary> Bollinger Bands Equation
</summary>  
<br>
<img src="img/equation_bollingerhigh.svg"><br>
<img src="img/equation_bollingerlow.svg"><br>       
where:<br>
    <i>σ</i> = standard deviation<br>
    <i>m</i> = number of standard deviations<br>
    <i>n</i> = number of days in the smoothing period<br>
</details>

---

# Machine Learning Model

LSTM (Long Short-Term Memory) model using TensorFlow and Keras is used. An example of the machine learning model code is provided in [<code>lstm_demo.ipynb</code>](code/ml/lstm_demo.ipynb) notebook.

## LSTM Overview
This application utilizes LSTM (Long Short-Term Memory) machine learning model. LSTM model was developed by Sepp Hochreiter and published in Neural Computation in 1997 [[Hochreiter 1997](https://dl.acm.org/doi/10.1162/neco.1997.9.8.1735)]. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell [Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory).

![lstm_cell](img/lstm_cell.png)

## Machine Learning Libraries

### TensorFlow

<img src="img/tf.png" width=200>

TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets developers easily build and deploy ML powered applications.

### Keras

<img src="img/keras.png" width=200>
 
Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library. Keras allows for easy implementation of TensorFlow methods without the need to build out complex machine learning infrastructure.

## Implementation
### Data Acquisition

Data is acquired from Alpaca Trade API and processed using the [<code>technicals</code>](code/technicals/technicals.py) module. The resulting DataFrame contains <code>Closing</code> price and all of the technical indicators. 

The market data is obtained by calling the <code>ohlcv()</code> method within the [<code>alpaca</code>](code/marketdata/alpaca.py) module. The methods takes a <code>list</code> of tickers, as well as the <code>start_data</code> and <code>end_date</code>, and returns a <code>pd.DataFrame</code>.

```python
end_date  = datetime.now().strftime('%Y-%m-%d')
start_date  = (end_date - timedelta(days=1000)).strftime('%Y-%m-%d')

ohlcv_df = alpaca.ohlcv(['tickers'], start_date=start_date, end_date=end_date)
```

The <code>TechnicalAnalysis</code> class must first be instantiated with the <code>pd.DataFrame</code> containing market data.

```python
tech_ind = technicals.TechnicalAnalysis(ohlcv_df)
tech_ind_df = tech_ind.get_all_technicals('ticker')
```

### LSTM model class

The LSTM model is contained within the <code>MachineLearningModel</code> class located in the [<code>lstm_model</code>](code/ml/lsmt_model.py) module. The class must first me instantiated with a <code>pd.DataFrame</code> containing the technical analysis data.

```python
my_model = lstm_model.MachineLearningModel(tech_ind_df)
```

### Build, fit and save model
Building and fitting the model is done by calling the <code>build_model()</code> class method.

```python
hist = my_model.build_model()
```

The model is then saved as an <code>.h5</code> file.

```python
my_model.save_model('model.h5')
```

## MachineLearningModel.build_model() Description
The <code>MachineLearningModel</code> is used to handle all machine learning methods. The <code>build_model()</code> class method, builds and fits the model. The class method implements the following methodology:

### Model overview
The LSTM model is programmed to look back <code>100</code> days to predict <code>14</code> days. The number of features is set by the shape of the DataFrame.

```python
n_steps_in = 100
n_steps_out = 14
n_features = tech_ind_df.shape[1]
```

### Scaling
A <code>RobustScaler</code> is used to scale the technical analysis data [[ScikitLearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)].

```python
sklearn.preprocessing.RobustScaler()
```

Scale features using statistics that are robust to outliers. 

This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile). Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Median and interquartile range are then stored to be used on later data using the transform method. 


### Parsing
The DataFrame is then parsed to <code>np.array</code> and spit into <code>X</code> and <code>y</code> subsets.

```python
X, y = split_sequence(tech_ind_df.to_numpy(), n_steps_in, n_steps_out)
```

Where <code>split_sequence()</code> is a helper method that splits the multivariate time sequences.

### Model type
<code>Sequential()</code> model is utilized as it groups a linear stack of layers into a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) [[TensorFlow]((https://www.tensorflow.org/api_docs/python/tf/keras/Sequential))]

```python
model = tf.keras.Sequential()
```

### Activation function
A hyperbolic tangent activation function is used : <code>tanh</code>[[TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh)]

```python
activation_function = tf.keras.activations.tanh
```

### Input and hidden layers
LSTM input and hidden layers are utilized. [[TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)]

The input layer contains <code>60</code> nodes, while the hidden layers contain <code>30</code> nodes by default but can be set by the administrator to *n* arbitrary amount by setting the <code>n_nodes</code> variable. The number of hidden layers default to <code>1</code> but can also be modified by the administrator.

Hidden layers are added with a <code>add_hidden_layers()</code> helper function.

```python
n_nodes = 30

# input layer
model.add(LSTM(60, 
               activation=activation_function, 
               return_sequences=True, 
               input_shape=(n_steps_in, n_features)))

# hidden layers ...
model.add(LSTM(n_nodes, activation=activation_function, return_sequences=True))
```

### Dense layers

Two dense layers are used in the model. Dense layers are added using <code>add_dense_layers</code> class method.

```python
model.add(Dense(30))
```

### Optimizer
The model uses Adam optimizer (short for Adaptive Moment Estimation) [[TensorFlow]((https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam))]. Adam is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments. Adam optimizer was developed by Diederik Kingma and Jimmy Ba and published in 2014 [[Kingma et. al. 2014](https://arxiv.org/pdf/1412.6980.pdf)]. Adam optimizer is defined by its creators as "an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments."

```python
optimizer = tf.keras.optimizers.Adam
```

### Loss function
The model uses Mean Squared Error loss function, which computes the mean of squares of errors between labels and predictions [[TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError)]

```python
loss = tf.keras.losses.MeanSquaredError
```

### Other model parameters
Model is trained for <code>16</code> epochs using <code>128</code> unit batch size. The validation split is <code>0.1</code>.


### Compiling and fitting

The model is then compiled and fit.

```python
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
hist = model.fit(X, y, epochs=16, batch_size=128, validation_split=0.1)
```

## Training Results

An example of model training results with conducted with The Coca-Cola Company stock : KO. 

### Accuracy

![model_accuracy_KO](img/model_accuracy_KO.png)

### Loss

![model_loss_KO](img/model_loss_KO.png)

### Predictions
Predictions are calculated with a <code>validator()</code> helper method.

![model_pred_KO](img/pred_prices_KO.png)


## Forecasting stock prices
### Implementation
To forecast stock prices using the saved model, the application uses the <code>ForecastPrice</code> class located within the [<code>lstm_model</code>](code/ml/lsmt_model.py) module.

The module pre-processes the date using the aforementioned methods and then utilizes <code.model.predict()</code> TensorFlow method.

The application accomplished this by:

1. Getting stock prices for past <code>200</code> days using <code>alpaca</code> module
2. Getting technical indicators using the <code>get_all_technicals()</code> method withing the <code>technicals.TechnicalAnalysis</code> class
3. Instantiating the <code>ForecastPrice</code> class with the technical data

```python
forecast_model = lstm_model.ForecastPrice(tech_ind_df)
```

4. Calling <code>forecast()</code> method within the <code>ForecastPrice</code> class

```python
forecast = forecast_model.forecast()
```

### ForecastPrice.forecast() Description

ForecastPrice class handles all of the forecasting functions. The <code>forecast()</code> class method implements the following methodology:

1. Load model using <code>load_model</code> Keras method.

```python
from tensorflow.keras.models import load_model
forecast_model = load_model("model.h5")
```

2. Pre-processes the data following the same methodology as MachineLearningModel class.

3. Predicts the prices.

```python
forecasted_price = forecast_model.predict(tech_ind_df)
```

4. Inverse scale the prices.

```python
forecasted_price = scaler.inverse_transform(forecasted_price)[0]
```


## Forecast Result

![model_pred_KO](img/model_forecast_KO.png)

If the predicted price <code>14</code> days from now is higher than the current price, the application will issue a buy "opinion", if the price is lower that the current price it will issue a sell "opinion" on the date of the highest predicted price.

---

# Sentiment Analysis
Sentiment analysis is performed using the [Google Cloud Natural Language](https://cloud.google.com/natural-language) service. 

![gc_nlp](img/gc_nlp.png)

The data utilized in sentiment analysis is obtained from 2 sources:

1. [NewsAPI](https://newsapi.org/)
2. [Tweepy](https://www.tweepy.org/)

Implementation of NewsAPI and Tweepy can be found in the demo notebook: <code>[nlp_demo.ipynb](code/nlp/nlp_demo.ipynb)</code>

The sentiment analysis implementation:

```python
from google.cloud import language_v1
from google.oauth2.credentials import Credentials

def GetSentimentAnalysisGoogle(text_content):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '../your_credentials_file.json'
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    document = {'content': text_content, 'type_': type_}
    encoding_type = language_v1.EncodingType.UTF8
    response = client.analyze_sentiment(request={'document': document, 
                                                 'encoding_type': encoding_type})
    return {'score' : response.document_sentiment.score , 
            'magnitude' : response.document_sentiment.magnitude}
```


---

# Team

* [Fernando Bastos](https://www.linkedin.com/in/fdobastos/)
* [Shaunjay Brown](https://www.linkedin.com/in/shaun-jay-brown-933b7437/)
* [Illya Nayshevsky, Ph.D.](http://www.illya.bio)
