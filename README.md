# Tweet Sentiment Analysis with Logistic Regression
This project implements sentiment analysis of tweets using a **Logistic Regression** classifier. It provides two main components:

- FastAPI API to serve the model and make predictions.
- A Streamlit app to interact with the API and visualize the results.

## Project Overview
The goal of this project is to classify tweets into two categories: **positive** or **negative** sentiment. The model used for classification is a **Logistic Regression** classifier, which is a traditional machine learning approach trained on pre-processed tweet data.

The project includes:

- **FastAPI** serves as the backend API for the model, allowing users to send GET requests for tweet sentiment predictions.
- **Streamlit** is used to create a simple user interface for interacting with the model via the API.

## Project Structure
```├── src/
│   ├── models/
│   │   └── LogisticRegressionClassifier.py  # Logistic regression model and its API
│   ├── api/
│   │   ├── api.py  # FastAPI app to serve the model
│   └── front-end/
│       └── app.py  # Streamlit app for front-end interface
└── requirements.txt  # Required dependencies
```

## Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/tweet-sentiment-analysis.git
cd tweet-sentiment-analysis
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

## Usage

### 1. Running the FastAPI server
To run the FastAPI backend and expose the model via API:

```
uvicorn src.api.api:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`. The prediction endpoint can be accessed with a `GET` request like::

```
http://localhost:8000/predict?tweet=hello
```

The API will respond with a sentiment classification (positive or negative).

### 2. Running the Streamlit app
To run the Streamlit app, which provides a simple UI for interacting with the model:

```
streamlit run src/front-end/app.py
```
The app will open in your browser and allow you to enter tweets, view predictions, and visualize the results.

# Model
**Logistic Regression**: A classical machine learning approach for sentiment classification. It has been trained on pre-processed tweet data and is fast and efficient for smaller datasets.
