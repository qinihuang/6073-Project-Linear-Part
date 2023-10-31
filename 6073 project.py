#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:29:26 2023

@author: huangqini
"""
import yfinance as yahooFinance
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('seaborn')

startDate = datetime.datetime(2010, 1, 1)
 
# endDate , as per our convenience we can modify
endDate = datetime.datetime(2023, 10, 30)
 
# pass the parameters as the taken dates for start and end
sp_400 = pd.DataFrame(yahooFinance.Ticker("^SP400").history(start=startDate, end=endDate))[['Close']]
sp_400 = sp_400.rename(columns={"Close": "sp_price"})
sp_400['sp_price'] = np.log(np.abs(sp_400['sp_price']))

rut_2000 = pd.DataFrame(yahooFinance.Ticker("^RUT").history(start=startDate, end=endDate))[['Close']]
rut_2000 = rut_2000.rename(columns={"Close": "rut_price"})
rut_2000['rut_price'] = np.log(np.abs(rut_2000['rut_price']))

df = sp_400.join(rut_2000)
model = sm.OLS(df['sp_price'],sm.add_constant(df['rut_price']))
results = model.fit()
results.summary()

#%% Create buy and sell signals and calculate returns
df['spread'] = results.resid
plt.figure(figsize = [10,5])
df['spread'].plot()

result = sm.tsa.stattools.adfuller(df['spread'])
print('ADF Sattistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

df['signal'] = 2*(df['spread'] < 0.3*df['spread'].std()).astype(int) - 1
df.index = pd.to_datetime(df.index)
df = df.sort_index()
plt.figure(figsize=[10,5])
plt.fill_between(df.index, df['signal'], 0, color = 'b')
plt.plot(df['spread'], color = 'r')
plt.show()

df['ret_signal'] = df['signal'].shift(1)*(df['spread']-df['spread'].shift(1))
plt.plot(df['ret_signal'].cumsum())
plt.ylabel('cumret')
plt.show()

#%% Model the spread as AR(1) process
model = sm.tsa.arima.ARIMA(df['spread'], trend = 'n', order = (1,0,0), missing = 'drop', enforce_stationarity = False)
model_fit = model.fit()
model_fit.summary()

print('The std of the error is %.4f' %np.sqrt(model_fit.mse))
df['signal_2'] = 2*(df['spread'] - df['spread'].shift(1)*model_fit.params[0] < np.sqrt(model_fit.mse)) - 1
df['ret_signal'] = df['signal_2'].shift(1)*(df['spread'] - df['spread'].shift(1))
df['ret_signal'].cumsum().plot()
plt.ylabel('cumret')
plt.show()

#%% Use another window to estimate hedge ratio to avoid looking forward bias
model = sm.OLS(df.iloc[0:1100,0], sm.add_constant(df.iloc[0:1100,1]))
results = model.fit()
results.summary()

df['spread'] = df.iloc[:,0] - results.params[0] - df.iloc[:,1]*results.params[1]
df1 = df.iloc[1100:,:]
plt.plot(df1['spread'])
result = sm.tsa.stattools.adfuller(df1['spread'])
df1['signal'] = 2*(df1['spread'] < df1['spread'].std()).astype(int) - 1
df1.index = pd.to_datetime(df1.index)
plt.fill_between(df1.index, df1['signal'],0, color = 'b')
plt.plot(df1['spread'],color = 'r')
plt.show()

df1['ret_signal'] = df1['signal'].shift(1)*(df1['spread'] - df1['spread'].shift(1))
df1['cum_ret'] = df1['ret_signal'].cumsum()
plt.plot(df1['cum_ret'])
plt.ylabel('cumret')

#%%
# Calculate daily returns
df1['daily_ret'] = df1['ret_signal']

# Calculate cumulative returns
df1['cum_ret'] = df1['daily_ret'].cumsum()

# Calculate total return
total_return = df1['cum_ret'][-1]

# Calculate APR (Annualized Percentage Rate)
days_in_year = 252  # Assuming 252 trading days in a year
cumulative_return = df1['cum_ret']
cumulative_return = cumulative_return[-1]
number_of_days = 2305  # Replace with the exact number of trading days in your dataset

# Calculate APR
apr = ((cumulative_return + 1)) ** (252 / number_of_days) - 1


# Calculate daily returns for risk metrics
daily_returns = df1['daily_ret']

# Calculate Sharpe Ratio
risk_free_rate = 0.03  # Change this to your preferred risk-free rate
sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(days_in_year)

# Calculate Maximum Drawdown and Maximum Drawdown Duration
cumulative_returns = df1['cum_ret']
roll_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - roll_max)
max_drawdown = drawdown.min()
max_drawdown_duration = len(drawdown[drawdown == max_drawdown])

# Create a table to display the results
results_table = pd.DataFrame({
    'Total Return': [total_return],
    'APR': [apr],
    'Sharpe Ratio': [sharpe_ratio],
    'Maximum Drawdown': [max_drawdown],
    'Maximum Drawdown Duration': [max_drawdown_duration]
})

# Print the results
print("Trading Strategy Performance Metrics:")
print(results_table)

#%%
import matplotlib.pyplot as plt

# Assuming you have dataframes 'sp_400', 'rut_2000', and 'df' containing the respective data
# Make sure you have calculated the pair trading strategy returns and stored them in df['ret_signal']

# Plot the returns of SP400 and RUT
plt.figure(figsize=(12, 6))
plt.plot(sp_400.index, sp_400['sp_price'].diff(), label='SP400 Returns', color='blue')
plt.plot(rut_2000.index, rut_2000['rut_price'].diff(), label='RUT Returns', color='green')

# Plot the pair trading strategy returns
plt.plot(df.index, df['ret_signal'], label='Pair Trading Returns', color='red')

plt.title('S&P 400, Russell 2000, and Pair Trading Strategy Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.show()





