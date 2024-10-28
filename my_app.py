
# cryptos ETH=1027, BTC = 1, XRP = 52, SOL = 5426, ADA = 2010
from predictor import analyze_crypto_data, analyze_crypto_data_api
from datetime import datetime, timedelta
from dotenv import load_dotenv, dotenv_values
import numpy as np
from data.my_database import create_database, csv_to_sqllite

import os
import requests
from flask_restful import Resource, Api
load_dotenv()

coinmarket_key = os.getenv("API_KEY")
url =  os.getenv("URL")
database = os.getenv("DATABASE")
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': coinmarket_key
}
# get today's date
today = datetime.now()


def get_crypto_values(value=1):
  crypto_values = {'1027': 'ETH', '1': 'BTC', '52': 'XRP'}
  for key, val in crypto_values.items():
    if val == value:
      return key, val
  return None

def get_parameters(key=1):
  parameters = {
    'id': key,
    'convert': 'USD',
    'time_start': today - timedelta(days=30),
    'time_end': today,
    'interval': 'daily',
    'count': '1',
  }
  return parameters 

'''parameters = {
  'id': 2,
  'convert': 'USD',
  'time_start': today - timedelta(days=30),
  'time_end': today,
  'interval': 'daily',
  'count': '1',
}
def show_predictions():
  predicted_price = analyze_crypto_data(url, headers, parameters)
  
  latest_prediction = 'price.png'

  return render_template('index.html', image_filename=latest_prediction, predicted_price = predicted_price)
'''

#analyze_crypto_data(url, headers, parameters)

from flask import Flask, render_template, send_from_directory, request
from flask_restful import Resource

my_app = Flask(__name__)
api = Api(my_app)
create_database(database)
class api_predictor(Resource):
   def get(self, key):
      
      
      value = get_crypto_values(key)
      parameters = get_parameters(value)
      
      predicted_price = analyze_crypto_data_api(url, headers, parameters, database)
      return {key: predicted_price}
   
api.add_resource(api_predictor, '/api_predictor/<string:key>')

@my_app.route('/')

def show_predictions():

  parameters = get_parameters()
  predicted_price = analyze_crypto_data(url, headers, parameters, database)
  key, val = get_crypto_values(predicted_price)
  latest_prediction = 'price.png'

  return render_template('index.html', image_filename=latest_prediction, predicted_price = predicted_price, key = val)



@my_app.route('/', methods=['POST'])

def crypto():
    if request.method == 'POST':
        predicted_price = request.form['cryptocurrency']
        
        key, val = get_crypto_values(predicted_price)
        parameters = get_parameters(key)
        predicted_price = analyze_crypto_data(url, headers, parameters, database)
        return render_template('index.html', predicted_price=predicted_price, key = val)
    else:
        return render_template('index.html')


# Serve static images
@my_app.route('/images/<filename>')
def serve_image(latest_prediction):
  return send_from_directory('/static/images/', latest_prediction)


      
#if __name__ == '__main__':

my_app.run(host='0.0.0.0', port=5000, debug=True)


       








