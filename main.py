
# cryptos ETH=1027, BTC = 1, XRP = 52, SOL = 5426, ADA = 2010
from analysis.predictor import analyze_crypto_data
from datetime import datetime, timedelta
coinmarket_key = '4cd46783-7c2c-4d25-96bc-b1793a2e899e'
url =  'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical'

headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': coinmarket_key
}
# get today's date
today = datetime.now()



def get_crypto_values(value):
  crypto_values = {'1027': 'ETH', '1': 'BTC', '52': 'XRP'}
  for key, val in crypto_values.items():
    if val == value:
      return key
  return None

def get_parameters(key):
  parameters = {
    'id': key,
    'convert': 'USD',
    'time_start': today - timedelta(days=30),
    'time_end': today,
    'interval': 'daily',
    'count': '1',
  }
  return parameters

parameters = {
  'id': 2,
  'convert': 'USD',
  'time_start': today - timedelta(days=30),
  'time_end': today,
  'interval': 'daily',
  'count': '1',
}

#analyze_crypto_data(url, headers, parameters)

from flask import Flask, render_template, send_from_directory, request

app = Flask(__name__)

@app.route('/')
def show_predictions():
  predicted_price = analyze_crypto_data(url, headers, parameters)
  latest_prediction = 'predictions.png'

  return render_template('index.html', image_filename=latest_prediction, predicted_price = predicted_price)

@app.route('/', methods=['POST'])
def crypto():
    if request.method == 'POST':
        predicted_price = request.form['cryptocurrency']
        key = get_crypto_values(predicted_price)
        parameters = get_parameters(key)
        predicted_price = analyze_crypto_data(url, headers, parameters)
        return render_template('index.html', predicted_price=predicted_price)
    else:
        return render_template('index.html')
'''def analyse():
  selected_crypto = request.form['cryptocurrency']
  crypto_value = crypto_values[selected_crypto]
  parameters['id'] = crypto_value
  return redirect(url_for('show_predictions'))'''


@app.route('/images/<filename>')
def serve_image(filename):
  return send_from_directory('/static/images/', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)




