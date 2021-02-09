import json
import plotly
import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify, session
from plotly.graph_objs import Bar, Scatter, Pie
from sqlalchemy import create_engine
import os
import json
from process_modules.load_data import LoadData
from process_modules.train_models import TrainModel
from move_data import load_models

# , template_folder='./templates',static_folder='./static'
app = Flask(__name__)
# create a secret key for session
app.secret_key = os.urandom(16)

# Read data from API and store in a db
load_data = LoadData('data/Stocks.db')
load_data.load()

# Load data
engine = create_engine('sqlite:///data/Stocks.db')
df = pd.read_sql_table('stocks', engine)

# Get options of stocks available to analyze
symbols = df.symbol.drop_duplicates().tolist()
symbols = [symbol for symbol in symbols if symbol != '^BVSP']

models_path = 'models/models.pkl'

# Target options
targets_dict = {'1_day': 1,
                '1_week': 5,
                '2_weeks': 5,
                '1_month': 22} 

# Load home page
@app.route('/')
@app.route('/index')
def index(symbols=symbols):
    return render_template('master.html', symbols=symbols)

# Load first query page
@app.route('/query')
def query(stocks_df=df, symbols=symbols, models_path=models_path, targets_dict=targets_dict):

    checked_symbols = []
    for i in range(len(symbols) + 1):
        if request.args.get("check" + str(i)) == 'on':
            checked_symbols.append(symbols[i-1])
    
    # Get start and end date from front end
    start_date = request.args.get("start-date")
    end_date = request.args.get("end-date")

    # Store data in session
    session['start_date'] = start_date
    session['end_date'] = end_date
    session['checked_symbols'] = checked_symbols 
    
    # Train all possible models for different targets
    train_model = TrainModel(
            stocks_df,
            checked_symbols,
            start_date,
            end_date,
            models_path,
            targets_dict)
    train_model.create_models()
    
    return render_template('query.html', checked_symbols=checked_symbols)

# Load query page to iteract n times
@app.route('/predict')
def predict(stocks_df=df, models_path=models_path, targets_dict=targets_dict):
    
    # Get session params
    checked_symbols = session.get('checked_symbols') 
    start_date = session.get('start_date')
    end_date = session.get('end_date') 

    # Get infos from front end
    forecast_date = request.args.get("forecast-date")
    target = request.args.get("target")
    symbol = request.args.get("symbol")

    # Convert dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    forecast_date = pd.to_datetime(forecast_date)

    # Prep model input vars for forecast date
    stock, ibov = TrainModel.prep_historical_data(
            stocks_df,
            symbol,
            forecast_date,
            forecast_date)
    vars_df = TrainModel.prep_modeling_data(
            stock,
            ibov,
            targets_dict,
            forecast_date,
            forecast_date,
            target_flag=False)
    
    # Load models
    models = load_models(models_path)

    # Recover specific model
    train_results = models[symbol][target]
    model = train_results['model']

    # Predict price for forecast date
    y_pred = model.predict(vars_df)[0]

    # Prepare data to plot
    train_data = TrainModel.get_data(stocks_df, symbol, start_date, end_date)
    pred_date = forecast_date + pd.DateOffset(targets_dict[target]) 
    pred_df = pd.DataFrame([[y_pred, pred_date]], columns=[symbol, 'date'])
    pred_df.set_index('date', inplace=True) 
    pred_line = pd.concat([train_data.tail(1), pred_df[symbol]])

    # Calculate results metrics for forecast date
    pred_date = pred_date.date()
    pred_price = round(y_pred, 2)
    current_price = round(vars_df[symbol][0], 2)
    price_var = round(((pred_price/current_price) - 1) * 100, 2)
    mae_test = train_results['mae_test']
    mae_test_pct = round((mae_test/pred_price) * 100, 2)

    results = {
            'pred_date': pred_date,
            'current_price': current_price,
            'pred_price': pred_price,
            'price_var': price_var,
            'mae_test_pct': mae_test_pct}

    # Create plot result
    trace1 = Scatter(
        x = list(train_data.index),
        y = list(train_data.values),
        mode = 'lines',
        name = 'historical_price'
    )

    trace2 = Scatter(
        x = list(pred_line.index),
        y = list(pred_line.values),
        mode = 'lines',
        name ='predicted_price'
    )
  
    layout = dict(title = symbol +' stock prices',
                xaxis = dict(title = 'Date'),
                yaxis = dict(title = 'Price (R$)'),
                width = 500,
                height = 300,
                margin = {
                    'l': 50,
                    'r': 50,
                    'b': 50,
                    't': 50,
                    'pad': 4}
    ) 

    data = []
    data.append(trace1)
    data.append(trace2)
    graph = dict(data=data, layout=layout)

    # encode plotly graphs in JSON
    graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('predict.html',
            results=results,
            graphJSON=graphJSON,
            checked_symbols=checked_symbols,
            target=target)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
