# finance-project
The majority of this project was developing a web app for calculating profit/loss for European and US style options.
This was developed using the Streamlit python library.
The app can be found here: optionpnl.streamlit.app

App.py is the main body and logic of the app. This script pulls functions from the Calculations.py and Data_Downloader.py files.
Data_Downloader.py has functions for retrieving commonly used financial data (e.g. bond yields, option chains, inflation etc.) from both web pages and modules such as Yfinance.
Calculations.py contains functions for solving the Black-Scholes equation both analytically and numerically, and for calculating implied volatility.

Monte Carlo.py is a Monte Carlo simulation for modelling the stochastic differential equation (SDE) for a stock price. It can plot a graph showing the expected range of prices, as well as providing various statistical meassures from the simulation. As with the App.py file, Monte Carlo.py uses functions from the Calculations.py and Data_Downloader.py files
