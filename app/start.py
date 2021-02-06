import json
import plotly
import pandas as pd
import pickle
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter, Pie
from sqlalchemy import create_engine
from flask import Flask
from process_modules.load_data import LoadData
from process_modules.train_models import TrainModel
from move_data import load_models

app = Flask(__name__, template_folder='./templates',static_folder='./static')

# Read from API
load_data = LoadData('data/Stocks.db')
load_data.load()

# Load data
engine = create_engine('sqlite:///data/Stocks.db')
df = pd.read_sql_table('stocks', engine)

symbols = df.symbol.drop_duplicates().tolist()
symbols = [symbol for symbol in symbols if symbol != '^BVSP']

models_path = 'models/models.pkl'

@app.route('/')
@app.route('/index')
def index(symbols=symbols):

    return render_template('master.html', symbols=symbols)

@app.route('/query')
def query(stocks_df=df, symbols=symbols, models_path=models_path):

    symbols_checked = []
    for i in range(len(symbols)):
        if request.args.get("check" + str(i)) == 'on':
            symbols_checked.append(symbols[i-1])
    
    start_date = request.args.get("start-date")
    end_date = request.args.get("end-date")
    
    train_model = TrainModel(stocks_df, symbols_checked, start_date, end_date, models_path)
    train_model.create_models()
    
    return render_template('query.html', symbols_checked=symbols_checked, train_model=train_model)

@app.route('/predict')
def predict(stocks_df=df, symbols=symbols, models_path=models_path):
    forecast_date = request.args.get("forecast_date")
    target = request.args.get("target")
    symbol = request.args.get("symbol")

    print(request.args.get("train_model"))
    models = load_models(models_path)
    train_results = models[symbol][target]
    model = train_results['model']
    

    return render_template('predict.html')

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
