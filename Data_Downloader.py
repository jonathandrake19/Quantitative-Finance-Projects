from bs4 import BeautifulSoup
import requests
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, date, timedelta

def bond_yield(country, duration): 
    if isinstance(duration, (int, float)):
        x = round(duration/30)
        duration = str(x) + 'M'

    url = 'https://www.investing.com/rates-bonds/' + country.lower() + '-government-bonds?maturity_from=40&maturity_to=310'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, features='lxml')

    table = soup.find_all('table')[0]
    titles = table.find_all('th')
    table_titles = [title.text.strip() for title in titles][1:-1]
    df = pd.DataFrame(columns = table_titles)

    column_data = table.find_all('tr')
    for row in column_data[1:]:
        row_data = row.find_all('td')
        individual_row_data = [data.text.strip() for data in row_data][1:-1]

        length = len(df)
        df.loc[length] = individual_row_data
    
    index = -1
    for row in df['Name']:
        index += 1
        if row.split(' ')[1] == duration:
            b_yield = df['Yield'][index]
    
    return float(b_yield)

def inflation():
    #if country.lower() == 'uk':
    url = 'https://www.ons.gov.uk/economy/inflationandpriceindices#timeseries'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, features= 'lxml')

    block = soup.find_all('div', class_='col col--md-13 col--lg-18')
    line = block[0].find('span')
    value = float([x.text.strip() for x in line][0])
    return value
    '''
    if country == 'usa':
        url = 'https://tradingeconomics.com/united-states/inflation-cpi'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, features= 'lxml')

        block = soup.find_all('div', class_= 'card')
        print(soup)
        '''           
def risk_free_rate(country, duration): #Returns a float
    return (bond_yield(country, duration) - inflation())/100

def stock_data(ticker):
    data = yf.download(ticker, start='2024-01-01', end=date.today())
    close_price = data['Close']
    log_return = np.log(1 + close_price.pct_change())
    return close_price, log_return

def option_data(ticker, expiry): #returns 2 pandas dataframes, 1 for calls, 1 for puts
    tk = yf.Ticker(ticker)

    options = pd.DataFrame()
    
    opt_chain = tk.option_chain(expiry)
    calls = opt_chain.calls
    puts = opt_chain.puts

    options = pd.concat([options, calls, puts], ignore_index = True)
    
    options = options.drop(columns= ['lastTradeDate', 'inTheMoney', 'contractSize', 'currency', 'percentChange', 'change', 
                                     'volume', 'impliedVolatility', 'bid', 'ask'])
    
    calls = pd.DataFrame(columns=options.columns)
    puts = pd.DataFrame(columns=options.columns)

    for _, row in options.iterrows():
        if 'c' in row['contractSymbol'][4:].lower():
            calls = pd.concat([calls, pd.DataFrame([row])], ignore_index = True)
        else:
            puts = pd.concat([puts, pd.DataFrame([row])], ignore_index = True)
    
    calls = calls.drop(columns = 'contractSymbol')
    puts = puts.drop(columns='contractSymbol')
    
    return calls, puts

def current_price(ticker):
    tk = yf.Ticker(ticker)
    S = (tk.history(period = '1d')['Close']).item()
    return S


