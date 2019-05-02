# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:57:41 2019

@author: jiaxi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


stock=pd.read_excel('telecom_stock.xlsx',sheetname='stock_price',parse_dates=['Dates'])
stock.set_index('Dates',inplace=True)
stock.head()

benchmark=pd.read_excel('telecom_stock.xlsx',sheetname='benchmark',parse_dates=['Dates'])
benchmark.set_index('Dates',inplace= True)
benchmark.head()

rf=pd.read_excel('telecom_stock.xlsx',sheetname='rf',parse_dates=['Dates'])
rf.set_index('Dates',inplace=True)
rf=rf['USGG10YR Index']/100/252


stock.plot(subplots=True, title='Telecom Stock')
stock.describe()

benchmark['SPX Index'].plot(subplots=False, title ='Benchmark')
benchmark.describe()

#Calculate daily returns

stock_returns=stock.pct_change()

stock_returns.describe()

sp_returns=benchmark.pct_change()

sp_returns.describe()

excess_returns=stock_returns.sub(rf,axis=0)  
excess_returns.describe()

avg_excess_return=excess_returns.mean()
sd_excess_return=excess_returns.std()

#Calculate Sharpe Ratio: SR = (Rp - Rf)/Std(Rp)

daily_sharpe_ratio=avg_excess_return.div(sd_excess_return)

annual_factor=np.sqrt(252)

annual_sharpe_ratio=daily_sharpe_ratio.mul(annual_factor)
annual_sharpe_ratio.plot.bar(title='Annualized Sharpe Ratio')

#company with max sharpe ratio
max_sr=annual_sharpe_ratio.idxmax()


#Calculate beta: beta(s)=Cov(s, market)/Var(market)
#Calculate beta: linear regression: Rp & Rm
#Calculate beta: CAPM Method: Rpt - RFt = alpha p + beta p[RMt - RFt] + ept



from sklearn.linear_model import LinearRegression

def linreg(x,y):
    model = LinearRegression().fit(x, y)
    alpha=model.intercept_
    beta= model.coef_
    return alpha, beta

df=pd.DataFrame(columns=['ticker','beta', 'alpha'])

for ticker in stock_returns.columns:
    
    alpha, beta= linreg((stock_returns[ticker]-rf).dropna().values.reshape(-1,1), (sp_returns['SPX Index']-rf).dropna().values)
    l=[pd.Series([ticker, beta,alpha],index=df.columns)]
    df=df.append(l,ignore_index=True)
    

df=df.set_index('ticker')
max_alpha=df['alpha'].idxmax()


