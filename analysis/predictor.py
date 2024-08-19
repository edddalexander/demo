import requests
import time
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
import numpy as np
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split    
from flask import Flask, render_template, request
import io  # Used for generating image data from plot
import matplotlib
import matplotlib.ticker as ticker
matplotlib.use('Agg')  # Required to save plots as static files
from matplotlib.backends.backend_agg import FigureCanvasAgg  # For embedding plots
import matplotlib.pyplot as plt  # Your existing plotting library


#coinmarket_key = API_KEY
url =  'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical'


'''headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': coinmarket_key,
}'''
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

def write_to_csv(crypto_data, filename='crypto_data.csv'):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'price', 'volume_24h']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(crypto_data)     
        
def process_crypto_data(crypto_data_file):
  test_data = pd.read_csv(crypto_data_file)  # Create DataFrame
  test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], utc=True)  # Convert timestamp to datetime
  test_data['timestamp'] = test_data['timestamp'].dt.date  # Extract date from datetime
  test_data['volume_24h'] = test_data['volume_24h'].apply(lambda x: f"{x:.2f}")  # Format volume_24h with 2 decimals
  test_data = test_data.rename(columns={'timestamp': 'date'})  # Rename timestamp column
  test_data['price'] = test_data['price'].apply(lambda x: f"{x:.2f}")  # Format price with 2 decimals
  test_data['future_price'] = test_data.price.shift(-1)  # Shift price one row down (using attribute access)
  test_data_na = test_data.dropna()
  return test_data, test_data_na



def prepare_and_train_linear_regression(test_data_na):
  # Separate features and target variable
  train_size = int(len(test_data_na) * 0.8)  # Adjust train size as needed
  train_data = test_data_na[:train_size]
  test_data = test_data_na[train_size:]
  # new data fram with just the date column test_data_date = test_data['date'] 
  test_data_date = test_data[['date']]

  
  train_data = train_data.iloc[1:]
  #test_data = test_data.iloc[1:]
  
  X_train = train_data[['price', 'volume_24h']]  # Select features
  y_train = train_data[ 'future_price']  # Target variable
  X_train = X_train.iloc[1:]
  y_train = y_train.iloc[1:]

  X_test = test_data[['price', 'volume_24h']]  # Select features
  y_test = test_data[ 'future_price']  # Target variable
  #X_test = X_test.iloc[1:]
  
  #y_test = y_test.iloc[1:]
  print(test_data_date)
  # Drop the first row (no corresponding target value)
  
  # Split data into training and testing sets (optional, can adjust test_size)
  #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
  
  # Extract dates for training and testing sets
  # Drop 'date' from X_train and X_test
  #X_train = X_train.drop('date', axis=1)
  #X_test = X_test.drop('date', axis=1)

  # Train the model
  model = LinearRegression()
  model.fit(X_train, y_train)


  # Make predictions on the testing set
  y_pred = model.predict(X_test)

  return y_test, y_pred, model, test_data_date

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
  
  plt.clf() 
  y_test = np.array(y_test, dtype=float)
  y_pred = np.array(y_pred, dtype=float)
  plt.scatter(y_test, y_pred)
  plt.title('Predicted vs Actual')
  plt.xlabel('Actual')
  plt.ylabel('Predicted')
  #plt.show()
  plt.savefig('static/images/predictions.png')
  
  #print(f"Predicted price for new data: {predicted_price}")

# function to plot actual and predicted prices as lines against date values


def plot_actual_and_predicted_prices(test_data_date, y_test, y_pred):
  plt.clf()
  plt.figure(figsize=(10, 6))
  print(type(y_test))
  y_test = np.array(y_test, dtype=float)
  y_pred = np.array(y_pred, dtype=float)
  #y_pred = pd.Series(y_pred)
  
  plt.plot(test_data_date['date'], y_test, label='Actual Price', color='blue')
  plt.plot(test_data_date['date'], y_pred, label='Predicted Price', color='red')
  plt.title('Predicted vs Actual')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
# Increase tick density for x-axis
  ax = plt.gca()  # Get current axis
  ax.xaxis.set_major_locator(ticker.AutoLocator())
  ax.yaxis.set_major_locator(ticker.AutoLocator())
  plt.xticks()
  plt.yticks()
  plt.grid
  plt.tight_layout()
  #plt.show()
  plt.savefig('static/images/price.png')
  



def analyze_crypto_data(url, headers, parameters):
  
  data = get_coinmarketcap(url, headers, parameters)
  
  crypto_data = extract_crypto_data(data)
  write_to_csv(crypto_data, filename='crypto_data.csv')
  test_data, test_data_na = process_crypto_data('crypto_data.csv')
  y_test, y_pred, model, test_data_date = prepare_and_train_linear_regression(test_data_na)

  predicted_price = predict_for_latest_data(test_data, model)
  plot_actual_and_predicted_prices(test_data_date, y_test, y_pred)
  plot_predictions(y_test, y_pred, predicted_price)
  return predicted_price 





#analyze_crypto_data(url, headers, parameters)

