import requests
import time
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split    
from flask import Flask, render_template, request
import io  # Used for generating image data from plot
from matplotlib.backends.backend_agg import FigureCanvasAgg  # For embedding plots
import matplotlib.pyplot as plt  # Your existing plotting library
coinmarket_key = '4cd46783-7c2c-4d25-96bc-b1793a2e899e'
url =  'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical'


headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': coinmarket_key,
}
# get today's date
today = datetime.now()
parameters = {
  'id': '1027',
  'convert': 'USD',
  'time_start': today - timedelta(days=30),
  'time_end': today,
  'interval': 'daily',
  'count': '1',
}

#class CurrenctyPredictor:
"""  def __init__(self):
     self.data = None
     self.crypto_data = None """

def get_coinmarketcap(url, headers, parameters):
      response = requests.get(url, headers=headers, params=parameters)
      data = response.json()
      return data
      
  
def extract_crypto_data(data):
  crypto_data = []
  for item in data['data']['quotes']:
    timestamp = item['timestamp']
    price = item['quote']['USD']['price']
    volume_24h = item['quote']['USD']['volume_24h']
    data_point = {"timestamp": timestamp, "price": price, "volume_24h": volume_24h}
    crypto_data.append(data_point)
  return crypto_data     
        
def process_crypto_data(crypto_data):
  test_data = pd.DataFrame(crypto_data)  # Create DataFrame
  test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], utc=True)  # Convert timestamp to datetime
  test_data['timestamp'] = test_data['timestamp'].dt.date  # Extract date from datetime
  test_data['volume_24h'] = test_data['volume_24h'].apply(lambda x: f"{x:.2f}")  # Format volume_24h with 2 decimals
  test_data = test_data.rename(columns={'timestamp': 'date'})  # Rename timestamp column
  test_data['price'] = test_data['price'].apply(lambda x: f"{x:.2f}")  # Format price with 2 decimals
  test_data['shifted_price'] = test_data.price.shift(-1)  # Shift price one row down (using attribute access)
  test_data_na = test_data.dropna()
  return test_data, test_data_na



def prepare_and_train_linear_regression(test_data_na):
  # Separate features and target variable
  X = test_data_na[['price', 'volume_24h']]  # Select features
  y = test_data_na['shifted_price']  # Target variable

  # Drop the first row (no corresponding target value)
  X = X.iloc[1:]
  y = y.iloc[1:]

  # Split data into training and testing sets (optional, can adjust test_size)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

  # Train the model
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Make predictions on the testing set
  y_pred = model.predict(X_test)

  return y_test, y_pred, model

def predict_for_latest_data(test_data, model):
 
  latest_values = test_data.iloc[-1, [1, 2]].to_list()

  # Define column names
  column_names = ['price', 'volume_24h']

  # Create a new DataFrame from the latest data point
  new_data = pd.DataFrame([latest_values], columns=column_names)

  # Predict the price for the new data point
  predicted_price = model.predict(new_data)[0]  # Access the first element of prediction


  return predicted_price

def plot_predictions(y_test, y_pred, predicted_price):
  
  plt.scatter(y_test, y_pred)
  plt.title('Predicted vs Actual')
  plt.xlabel('Actual')
  plt.ylabel('Predicted')
  #plt.show()
  plt.savefig('static/images/predictions.png')
  #print(f"Predicted price for new data: {predicted_price}")

def analyze_crypto_data(url, headers, parameters):
  data = get_coinmarketcap(url, headers, parameters)
  crypto_data = extract_crypto_data(data)
  test_data, test_data_na = process_crypto_data(crypto_data)
  y_test, y_pred, model = prepare_and_train_linear_regression(test_data_na)

  predicted_price = predict_for_latest_data(test_data, model)
  plot_predictions(y_test, y_pred, predicted_price)
  return predicted_price

analyze_crypto_data(url, headers, parameters)

