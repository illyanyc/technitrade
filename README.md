![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&width=1000&height=200&section=header&text=Project%202&fontSize=30&fontColor=black)
<!-- header is made with: https://github.com/kyechan99/capsule-render -->
Columbia FinTech Bootcamp

---

### Table of Contents
* [Overview](#overview)
* [Installation](#intallation)
* [Data](#data)
* [AWS Interface](#aws-interface)
* [Technical Analysis](#technical-analysis)
* [Sentiment Analysis](#sentiment-analysis)
* [Team](#team)

---
# Overview

Get stock alerts based on technical analysis and sentiment analysis powered by AI. Using an AWS chatbot, the user can track up to 10 stocks (or 10 AI selected stocks), and receive buy, sell or hold recommendations. The recommendations are based on LSTM models trained on a multitude of technical indicators and sentiment analysis models using NewsAPI articles and Twitter. The recommendations allow the user to find optimal position entry and exit points.

---

# Installation

---

# Data

* Database - AWS RDS
    1. GUID
    2. Name
    3. Start date
    4. 10 stocks - tickers


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

* Using Backtrader
* Data from Yahoo Finance
* Indicators: 
    1. Relative Strength Index
    2. Relative Momentum Index
    3. EMA Cross
    4. Williams %R Indicator
    5. Ultra Indicator
    6. Money Flow Indicator
* Calculate indicators
* Calculate signals
* Train LSTM on signals
* Fit model with new data daily provide user with probability of returns

---

# Sentiment Analysis

* Natural Language Processing data source:
    1. NewsAPI 
    2. Twitter
* Google Cloud Natural Language - Pre-trained model
    * Get sentiment
* Fit model with new data daily provide user with probability of returns

---

# Team

* [Fernando Bastos](https://www.linkedin.com/in/fdobastos/)
* [Shaunjay Brown](https://www.linkedin.com/in/shaun-jay-brown-933b7437/)
* [Illya Nayshevsky, Ph.D.](http://www.illya.bio)
* [Fabio Reato](https://www.linkedin.com/in/fabio-reato-0a5086147/)
