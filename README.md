# Stock Prices Predictor Project
An initial approach to predict brazilian stock prices
### Table of Contents

1. [Installation and Instructions](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation and Instructions <a name="installation"></a>

All aditional python libraries to run this project's scripts are in *requirements.txt*. The code should run in Python versions 3.*.<p />
1. To install all dependences, run in comand line:<br />
          `pip install -r requirements.txt`

2. To run Analysis.ipynb is necessary to install jupyter notebook, with the following command:
    `pip install jupyter notebook`

3. Run the following command in the project's root directory to run web app:
    `python app/start.py`

4. Go to http://localhost:3001/. It can take some time to start, because the first step is reading data from API.


## Project Motivation<a name="motivation"></a>

For this project, I was interestested to study more about stock market and time series. Moreover, I was eager to know
which kind of features I will use and check how well I will predict stocks in a first approach.

All data are from [Yahoo Finance](https://finance.yahoo.com/) and extracted by [yahooquery](https://pypi.org/project/yahooquery/) API. 

## File Descriptions <a name="files"></a>

1. File structure of the project:

<pre>
<code>
.
|-- app
|   |-- static
|   |   | styles.css # pages styling 
|   |-- templates
|   |   |-- master.html  # main page of web app
|   |   |-- predict.html # Web page for query iteratively predict results
|   |   |-- query.html # Web page for query first time for predict results
|   |-- start.py  # Flask file that runs app
|-- data
|-- docs
|   |-- imgs
|   |   |-- 
|-- models 
|-- pkgs
|   |-- feat_metrics #  package for create features
|   |   |-- __init__.py # package init
|   |   |-- feat_metrics.py # module containing metric calculation functions
|   |-- move_data #  package for move data
|   |   |-- __init__.py # package init
|   |   |-- move_data.py # module containing data moving functions
|   |-- process_modules # Package for run pipeline functions
|   |   |-- __init__.py # package init
|   |   |-- load_data.py # module containg functions with all pipeline to read data from API and save in db
|   |   |-- train_models.py # module containg functions with pipelines to prep datasets, train models and score results
|   |-- setup.py # package setup
|-- Analysis.ipynb # Analysis and discussions about the problem
|-- LICENSE 
|-- README.md
|-- requirements.txt # dependencies


</code>
</pre>

2. `data/Stocks.db` will be created after start app with 1 table:
- *stocks* - historical stock prices read from API

2.1 For now, it is loading only 9 tickers from API:
- ^BVSP (IBOVESPA - market index)
- ITSA4 (Itaúsa - Holding with Itau, havaianas, duratex and other brands)
- MGLU3 (Magazine Luiza - one of the biggest retail company in Brazil)
- VVAR3 (Via Varejo - another big retail company in Brazil)
- WEGE3 (WEG - big company for industrial and energetic solutions)
- MDIA3 (MediaCo - big company for food sector)
- LREN3 (Lojas Renner - large clothing retail store)
- ITUB3 (Itau - biggest Latim America private bank) 
- EGIE3 (Engie Brasil Energia - Big private company in electrical sector)

3. `models/models.pkl` will be created after choose tickers and train period do fit the model

3.1 target times are predefined according forecast date (baseground date) choosen:
- `1_day` - 1 next workday after forecast date
- `1_week` - 5 workdays after forecast date
- `2_weeks` - 10 workdays after forecast date
- `1_month` - 22 workdays after forecast date

## Results <a name="results"></a>

1. Home app:

2. Example of model's predict:

3. Plot results:

4. Features used to train the model:

5. Modeling challeges:


## Licensing, Authors, Acknowledgements <a name="licensing"></a>

All data credit is from Yahoo Finance and data extraction from yahooquery. Otherwise, use the code as you wish. 
