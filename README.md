![header](img/logo.svg)


---
**Disclosure**

:warning: NOT INVESTMENT ADVICE :warning:

The content produced by this application is for informational purposes only, you should not construe any such information or other material as legal, tax, investment, financial, or other advice. Nothing contained in this article, Git Repo or withing the output produced by this application constitutes a solicitation, recommendation, endorsement, or offer by any member involved working on this project, any company they represent or any third party service provider to buy or sell any securities or other financial instruments in this or in in any other jurisdiction in which such solicitation or offer would be unlawful under the securities laws of such jurisdiction. This appliation was created solely to satisfy the requirements of Columbia University FinTech Bootcamp Project #2 Homework.

The use of word "recommendation" in this article, withing the application, or within information produced by the application is for demonstration purposes only, and is not a recommendation to to buy or sell any securities or other financial instruments!

---

### Table of Contents
* [Overview](#overview)
* [Application Logic](#application-logic)
* [Libraries](#libraries)
* [AWS Interface](#aws-interface)
* [Technical Analysis](#technical-analysis)
* [Machine Learning Model](#machine-learning)
* [Sentiment Analysis](#sentiment-analysis)
* [Team](#team)

---
# Overview

Technitrade lets user track a portfolio of stocks, periodically getting buy, sell, or hold recommendations based on analysis performed by machine learning models and investor sentiment calculated by natural langualge processing analysis of recent news articles and Tweets.

The user interacts with the program via an [Amazon Lex chatbot](#aws-interface). The machine learning analysis is performed using [LSTM (Long Short-Term Memory) model](#machine-learning). The model is trained on [technical analysis indicators](#technical-analysis). Sentiment analysis is performed by [Google Cloud Natural Language](#sentiment-analysis) using NewsAPI and Twitter APIs as data source.

---

# Application Logic

![flowchart](img/flowchart.svg)

---

# Libraries

The following libraries are used:

### Data and Computation
* [Numpy](https://numpy.org/) - "The fundamental package for scientific computing with Python"
* [Pandas](https://pandas.pydata.org/) - data analysis and manipulation tool

### Data Source APIs
* [Alpaca Trade API](https://alpaca.markets/docs/) - Internet brokerage
* [NewsAPI](https://newsapi.org/) - NewsAPI locates articles and breaking news headlines from news sources and blogs across the web and returns them as JSON.
* [Twitter API](https://developer.twitter.com/en/docs) - Twitter API enables programmatic access to Twitter.

### Machine Learning
* [Tensorflow](https://www.tensorflow.org/) - an end-to-end open source platform for machine learning.
* [Keras](https://keras.io/) - a deep learning API for Python


---

# AWS Interface

1. User provides the service with 10 stocks to track.
2. The service prepares and trains machine learning models for the stock data.
    a. Technical Indicator Backtests 
    b. Sentiment Analysis - Natural Language Processing
3. The service provides user with daily updates about the positions
    a. Buy, sell or hold recommendation
    b. What if scenario - “What if the user bought and held the positions”

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


### Stoichastic Oscillator

The stochastic oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time. The sensitivity of the oscillator to market movements is reducible by adjusting that time period or by taking a moving average of the result. It is used to generate overbought and oversold trading signals, utilizing a 0–100 bounded range of values. [[Investopedia](https://www.investopedia.com/terms/s/stochasticoscillator.asp)]

<details>
<summary> Stoichastic Oscillator Equation
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


---

# Sentiment Analysis

* Natural Language Processing data source:
    1. NewsAPI 
    2. Twitter
* Google Cloud Natural Language - Pre-trained model
    * Get sentiment
    * https://cloud.google.com/natural-language/docs/analyzing-sentiment
* Fit model with new data daily provide user with probability of returns



---

# Team

* [Fernando Bastos](https://www.linkedin.com/in/fdobastos/)
* [Shaunjay Brown](https://www.linkedin.com/in/shaun-jay-brown-933b7437/)
* [Illya Nayshevsky, Ph.D.](http://www.illya.bio)
