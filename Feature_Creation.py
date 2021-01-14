#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def RSI(df,window):
    """
    Relative Strength index of a ETF
    """
    # Window length for moving average
    window_length = window

    # Dates
    start = '2014-02-31'
    end = '2019-12-31'

    # Get data
    data = df
    # Get just the adjusted close
    close = data['Adj Close']
    # Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous 
    # row to calculate the differences
    delta = delta[1:] 

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = up.ewm(span=window_length).mean()
    roll_down1 = down.abs().ewm(span=window_length).mean()

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    # Calculate the SMA
    #roll_up2 = up.rolling(window_length).mean()
    #roll_down2 = down.abs().rolling(window_length).mean()

    # Calculate the RSI based on SMA
    #RS2 = roll_up2 / roll_down2
    #RSI2 = 100.0 - (100.0 / (1.0 + RS2))

    # Compare graphically
    #plt.figure(figsize=(8, 6))
    #RSI1.plot()
    #RSI2.plot()
    #plt.legend(['RSI via EWMA', 'RSI via SMA'])
    #plt.show()
    df['RSI'] = RSI1
    return df


# In[3]:


def MACD_mod(df,nl=12,nh=26,nsig=9):
    """
    Moving Average Convergence Divergence of an ETF compared to its signal line
    """
    # Get just the adjusted close
    close = df['Adj Close']
    mal = close.ewm(span=nl).mean()
    mah = close.ewm(span=nh).mean()
    macd = mal-mah
    sig = macd.ewm(span=nsig).mean()
    
    df['MACD'] = macd-sig
    return df


# In[1]:


def create_features(df,rsi_window = 14,macd_feat = [12,26,9]):
    """
    Takes in a dataframe of ETF and creates features with indicators
    """
    df.dropna(inplace=True)
    ## day and month
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['dayowk'] = df['Date'].dt.dayofweek
    df = pd.get_dummies(data = df,columns = ['Month','dayowk'])
    
    ##Previos n-day pct_changes
    df['1day_pct'] = df['Adj Close'].pct_change()
    df['2day_pct'] = df['Adj Close'].pct_change(periods = 2)
    df['3day_pct'] = df['Adj Close'].pct_change(periods = 3)
    df['4day_pct'] = df['Adj Close'].pct_change(periods = 4)
    df['5day_pct'] = df['Adj Close'].pct_change(periods = 5)
    df['7day_pct'] = df['Adj Close'].pct_change(periods = 7)
    
    ##Cumulative sum of 1day_pct
    df['1day_pct_cs'] = df['Adj Close'].pct_change().cumsum()
    
    ##EWMA of 7, 50 and 200 days
    df['ewma_7'] = df['Adj Close'].ewm(span=7).mean()/df['Adj Close']
    df['ewma_50'] = df['Adj Close'].ewm(span=50).mean()/df['Adj Close']
    df['ewma_200'] = df['Adj Close'].ewm(span=200).mean()/df['Adj Close']
    ## Golden Cross vs Death Cross etc.
    #df['7g(50&200)'] = (df['ewma_7'] > df['ewma_50']) & (df['ewma_7'] > df['ewma_200'])
    #df['7l(50&200)'] = (df['ewma_7'] < df['ewma_50']) & (df['ewma_7'] < df['ewma_200'])
    #df['7g50'] = (df['ewma_7'] > df['ewma_50']) & (df['ewma_7'] < df['ewma_200'])
    #df['7g200'] = (df['ewma_7'] < df['ewma_50']) & (df['ewma_7'] > df['ewma_200'])
    
    ##RSI and MACD
    df = RSI(df,14)
    df = MACD_mod(df,nl=macd_feat[0],nh=macd_feat[1],nsig=macd_feat[2])
    
    df['day_var'] = (df['High'] - df['Low'])/df['Close']## Days variance
    df['open_close'] = (df['Open'] - df['Close'])/df['Close'] ## Days Open-Close
    df['high_close'] = (df['High'] - df['Close'])/df['Close'] ##Days High-Close
    df['open_prev_close'] = (df['Open'] - df['Close'].shift(1))/df['Close'] ## Days open - Previos Dyas Close
    
    ##Classification target
    df['target'] = round((np.sign(df['1day_pct']).shift(-1)+1)/2) ## Target for classification
    #df['1_day_target'] =  df['Adj Close'].shift(-1) - df['Adj Close'] ## Target for Regression
    #df['target2'] = round((np.sign(df['1day_pct']).shift(-1)+1)/2)## Will the price go up intra-day
    
    ## IS the stock Overbought or Oversold based on RSI?
    df['RSI_overbought'] = df['RSI']>70
    df['RSI_oversold'] = df['RSI']<30
    
    
    #df.drop(['Open','High','Low','Close'],axis=1,inplace=True)
#    df = df.dropna()
    
    #df = df.reset_index(drop=True)
    
    ## Calculating how large the previos hot and cold streaks were
    f = 0
    df['prev_hot_streak'] = np.zeros(df.shape[0])
    for i in range(df.shape[0]-1):
        if df['target'][i] ==1:
            f += 1
            if df['target'][i+1] ==0:
                df['prev_hot_streak'][i+1] = f
                f = 0
    for i in range(1,df.shape[0]):
        #print(i)
        if  df['prev_hot_streak'][i]==0:
            df['prev_hot_streak'][i]=df['prev_hot_streak'][i-1]
    
    
    df['prev_cold_streak'] = np.zeros(df.shape[0])
    for i in range(df.shape[0]-1):
        if df['target'][i] ==0:
            f += 1
            if df['target'][i+1] ==1:
                df['prev_cold_streak'][i+1] = f
                f = 0

    for i in range(1,df.shape[0]):
        #print(i)
        if  df['prev_cold_streak'][i]==0:
            df['prev_cold_streak'][i] = df['prev_cold_streak'][i-1]
    
    ## Calculating current hot and cold streaks
    df['current_hot_streak'] = np.zeros(df.shape[0])
    df['current_cold_streak'] = np.zeros(df.shape[0])
    fhot=0
    fcold=0
    for i in range(df.shape[0]):
        if df['target'][i]==1:
            fhot += 1
            fcold = 0
            df['current_hot_streak'][i] = fhot
        elif df['target'][i]==0:
            fcold += 1
            fhot = 0
            df['current_cold_streak'][i] = fcold
    
    df['prev_hot_streak'] = df['prev_hot_streak'].shift(1)
    df['prev_cold_streak'] = df['prev_cold_streak'].shift(1)
    df['current_hot_streak'] = df['current_hot_streak'].shift(1)
    df['current_cold_streak'] = df['current_cold_streak'].shift(1)
    
    ## Combinations of previos streaks
    df['prev_current_hot'] = df['prev_hot_streak'] - df['current_hot_streak']
    df['prev_current_cold'] = df['prev_cold_streak'] - df['current_cold_streak']
    df['current_hot_prev_cold'] = df['current_hot_streak'] - df['prev_cold_streak']
    df['current_cold_prev_hot'] = df['current_cold_streak'] - df['prev_hot_streak']
    
    
    
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    return df
    


# In[ ]:




