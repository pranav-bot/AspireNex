# Deep Business Analysis of Telco Customer Churn using ML

![Telco Customer Churn](telco_customer_churn.jpg)

## Overview

This project explores Telco customer churn prediction using machine learning techniques. The goal is to analyze customer behavior and demographics to predict whether a customer is likely to churn (leave the service provider). The analysis includes deploying multiple classification models and determining the best-performing model for predicting customer churn.

## Contents

1. [Project Background](#project-background)
2. [Data](#data)
3. [Models Deployed](#models-deployed)
4. [Streamlit App for SMS Spam Prediction](#streamlit-app-for-sms-spam-prediction)
5. [Results](#results)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

## Project Background

Customer churn is a critical metric for businesses, especially in subscription-based services like telecom. Predicting churn can help businesses take proactive measures to retain customers, thereby improving customer satisfaction and reducing revenue loss.

This project utilizes machine learning algorithms to analyze historical customer data from Telco. Various features such as customer demographics, services subscribed, and billing information are used to predict whether a customer is likely to churn.

## Data

The dataset used for this analysis is sourced from Telco and contains information on customer attributes, services subscribed, contract details, and churn status. It includes:

- **Features**: Customer demographics (age, gender), services subscribed (internet, phone), contract details (tenure, payment method), etc.
- **Target**: Churn status (Yes/No)

## Models Deployed

Several machine learning models are deployed and evaluated for predicting customer churn. The models include:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (SVC)
- Gradient Boosting Classifier
- XGBoost Classifier
- Neural Network (Deep Learning) models

The performance of each model is evaluated based on metrics such as accuracy, precision, recall, and F1-score.

## SMS Spam Prediction using Machine Learning and Streamlit

#Overview

This project predicts whether an SMS message is spam or not using various machine learning classification models. It includes a Streamlit web application that allows users to input an SMS message and view the prediction from deployed models.
Contents

  -Project Background
  -Dataset
  -Models Deployed
  -Streamlit App
  -Results
  -Usage
  -Contributing
  -License

# Project Background

SMS spam detection is crucial for mobile users to avoid fraudulent or unwanted messages. This project uses natural language processing (NLP) techniques and machine learning algorithms to classify SMS messages as spam or not spam. Various models are evaluated to determine the most accurate predictor.
Dataset

The dataset used for training and evaluation consists of labeled SMS messages, where each message is classified as either spam or not spam. It includes:

    Features: Text content of SMS messages
    Target: Binary labels indicating spam (1) or not spam (0)
    
https://mpb3fruqc5se6dajxaiglz.streamlit.app/
