# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:59:17 2023

@author: kisah
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime
import matplotlib.pyplot as plt



## Create supertrend indicator for strategy
def generateSupertrend(df,close_array, high_array, low_array, atr_period, atr_multiplier):

    ## Truerange calculation for ATR
    df['TR1'] = abs(df['High'] - df['Close'].shift(1))
    df['TR2'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR3'] = abs(df['High'] - df['Low'])
    df['TrueRange'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    
    ## applied SMA ATR
    df['TrueRange'] = df['TrueRange'].rolling(atr_period).mean()
    

    ## initialize necessary values
    previous_final_upperband = 0
    previous_final_lowerband = 0
    final_upperband = 0
    final_lowerband = 0
    previous_close = 0
    previous_supertrend = 0
    supertrend = []
    supertrendc = 0

    ## calculation of Supertrend
    for i in range(0, len(close_array)):
        if np.isnan(close_array[i]):
            pass
        else:
            highc = high_array[i]
            lowc = low_array[i]
            atrc = df.loc[i,'TrueRange']
            closec = close_array[i]

            if math.isnan(atrc):
                atrc = 0

            basic_upperband = (highc + lowc + closec) / 3 + atr_multiplier * atrc
            basic_lowerband = (highc + lowc + closec) / 3 - atr_multiplier * atrc

            if basic_upperband < previous_final_upperband or previous_close > previous_final_upperband:
                final_upperband = basic_upperband
            else:
                final_upperband = previous_final_upperband

            if basic_lowerband > previous_final_lowerband or previous_close < previous_final_lowerband:
                final_lowerband = basic_lowerband
            else:
                final_lowerband = previous_final_lowerband

            if previous_supertrend == previous_final_upperband and closec <= final_upperband:
                supertrendc = final_upperband
            else:
                if previous_supertrend == previous_final_upperband and closec >= final_upperband:
                    supertrendc = final_lowerband
                else:
                    if previous_supertrend == previous_final_lowerband and closec >= final_lowerband:
                        supertrendc = final_lowerband
                    elif previous_supertrend == previous_final_lowerband and closec <= final_lowerband:
                        supertrendc = final_upperband

            supertrend.append(supertrendc)

            previous_close = closec

            previous_final_upperband = final_upperband

            previous_final_lowerband = final_lowerband

            previous_supertrend = supertrendc
       
    ## drop unnecessary columns, add supertrend as column, drop first 50 data for unbalanced ATR
    df = df.drop(columns=['TR1','TR2','TR3','TrueRange'])
    df['Supertrend'] = supertrend
    df = df.drop(index=df.head(50))
    df = df.reset_index(drop=True)
    return df


## backtest of data
## df is market data
def backtest(df):

    df['order_type'] = 0
    df['Return_strategy'] = 0
    
    ## creating list for order datas.
    buy_open_time = []
    buy_open_price = []
    sell_open_time = []
    sell_open_price = []

    buy_close_time = []
    buy_close_price = []
    sell_close_time = []
    sell_close_price = []
    
    
    order_holder = 'buy'
    stop_loss_hit = 0
    
    if(df.loc[0,'Supertrend'] > df.loc[0,'Open']):
        order_holder = 'sell'
        
    df.loc[0,'order_type'] = order_holder
    
    
    if(order_holder == 'sell'):
        df.loc[0,'Return_strategy'] = (df.loc[0,'Close'] - df.loc[1,'Close'])/df.loc[0,'Close']
        sell_open_price.append(df.at[0,'Open'])
        sell_open_time.append(df.at[0,'Timestamp'])
    else:
        df.loc[0,'Return_strategy'] = (df.loc[1,'Close'] - df.loc[0,'Close'])/df.loc[0,'Close']
        buy_open_price.append(df.at[0,'Open'])
        buy_open_time.append(df.at[0,'Timestamp'])
        
        
    
    ## finding entry points and exit points for the strategy.
    for index,row in df.iterrows():
        if(index == 0):
            continue
        
        df.at[index,'order_type'] = order_holder
        
        if(df.at[index - 1,'Supertrend'] < df.at[index - 1,'Close'] and df.at[index,'Supertrend'] > df.at[index,'Close']):
            if(stop_loss_hit == 0):
                buy_close_time.append(df.at[index - 1,'Timestamp'])
                buy_close_price.append(df.at[index - 1,'Supertrend'])
            
            stop_loss_hit = 0
            df.at[index,'order_type'] = 'wait'
            order_holder = 'sell'
            sell_open_time.append(df.at[index + 1,'Timestamp'])
            sell_open_price.append(df.at[index + 1,'Open'])
            
        if(df.at[index - 1,'Supertrend'] > df.at[index - 1,'Close'] and df.at[index,'Supertrend'] < df.at[index,'Close']):
            if(stop_loss_hit == 0):
                sell_close_time.append(df.at[index - 1,'Timestamp'])
                sell_close_price.append(df.at[index - 1,'Supertrend'])
                
            stop_loss_hit = 0
            df.at[index,'order_type'] = 'wait'
            order_holder = 'buy'
            buy_open_time.append(df.at[index + 1,'Timestamp'])
            buy_open_price.append(df.at[index + 1,'Open'])
            
        if(order_holder == 'buy' and stop_loss_hit == 0 and df.at[index,'Supertrend'] > df.at[index,'Low'] and df.at[index,'order_type'] != 'wait'):
            buy_close_time.append(df.at[index,'Timestamp'])
            buy_close_price.append(df.at[index,'Supertrend'])
            stop_loss_hit = 1
            
        if(order_holder == 'sell' and stop_loss_hit == 0 and df.at[index,'Supertrend'] < df.at[index,'High'] and df.at[index,'order_type'] != 'wait'):
            sell_close_time.append(df.at[index,'Timestamp'])
            sell_close_price.append(df.at[index,'Supertrend'])
            stop_loss_hit = 1
            

       

    ## close last order of market
    if (len(buy_open_price) - 1 == len(buy_close_price)):
        buy_close_price.append(df.at[index,'Close'])
        buy_close_time.append(df.at[index,'Timestamp'])
        
        
    if (len(sell_open_price) - 1 == len(sell_close_price)):
        sell_close_price.append(df.at[index,'Close'])
        sell_close_time.append(df.at[index,'Timestamp'])
        
        
    ## creating order data
    buy_df = pd.DataFrame()
    buy_df['Open_price'] = buy_open_price
    buy_df['Open_time'] = buy_open_time
    buy_df['Close_price'] = buy_close_price
    buy_df['Close_time'] = buy_close_time
    
    
    sell_df = pd.DataFrame()
    sell_df['Open_price'] = sell_open_price
    sell_df['Open_time'] = sell_open_time
    sell_df['Close_price'] = sell_close_price
    sell_df['Close_time'] = sell_close_time
    
    buy_df['Return'] = (buy_df['Close_price'] - buy_df['Open_price'])/buy_df['Open_price']
    sell_df['Return'] = (sell_df['Open_price'] - sell_df['Close_price'])/sell_df['Open_price']
    
    return buy_df,sell_df



def trailing_hit_check (df,buy_df,sell_df,percentage):
    
    buy_df[str(percentage) + 'return'] = False
    sell_df[str(percentage) + 'return'] = False
    
    for i in range(0,len(buy_df)):
        start_index = df[df['Timestamp'] == buy_df.at[i,'Open_time']].index
        end_index = df[df['Timestamp'] == buy_df.at[i,'Close_time']].index
        start_index = start_index[0].item()
        end_index = end_index[0].item()
        
        for index in range(start_index, end_index):
            if(df.at[index,'High'] > buy_df.at[i,'Open_price'] * (1 + percentage/100)):
                buy_df.at[i,str(percentage) + 'return'] = True
                break
                
    for i in range(0,len(sell_df)):
        start_index = df[df['Timestamp'] == sell_df.at[i,'Open_time']].index
        end_index = df[df['Timestamp'] == sell_df.at[i,'Close_time']].index
        start_index = start_index[0].item()
        end_index = end_index[0].item()
        
        for index in range(start_index, end_index):
            if(df.at[index,'Low'] < sell_df.at[i,'Open_price'] * (1 - percentage/100)):
                sell_df.at[i,str(percentage) + 'return'] = True
                break
    
    return buy_df,sell_df

symbol_list = ['AVAXUSDT','BTCUSDT','ETHUSDT','FILUSDT','GRTUSDT','LINKUSDT','MATICUSDT','SOLUSDT','XLMUSDT','CFXUSDT','BNBUSDT']
symbol_list = ['BTCUSDT']
dataframe_list = ['15']

for symbol in symbol_list:
    for dataframe in dataframe_list:
        
        col_names = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore']
        df = pd.read_csv('C:/Users/kisah/Desktop/Crypto_Data/' + dataframe + 'm_future/' + symbol + dataframe + 'MINUTE2020-2023.csv',names=col_names, header=None)
        df['Date'] = df['Timestamp'].apply(lambda x: datetime.fromtimestamp(x / 1000))
        df['Timestamp'] = df['Date']
        df = df.drop(columns = ['Date','closeTime','quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
        df.drop(index=df.index[0], axis=0, inplace=True)
        df = df.reset_index(drop=True)
        
        
        ## period list and multiplier list for finding optimum point of strategy.
        period_list = [10,15,20,25]
        multiplier_list = [2.5,3,3.5,4]

        ## trailing take profit points.
        percentage_list = np.arange(1, 6, step=1)
        
        return_df_buy = pd.DataFrame()
        return_df_sell = pd.DataFrame()
        return_df_buy_trailing_tp = pd.DataFrame()
        return_df_sell_trailing_tp = pd.DataFrame()

        for period in period_list:
            for multiplier in multiplier_list:
                
                df_temp = generateSupertrend(df.copy(), df['Close'].copy(), df['High'].copy(), df['Low'].copy(), period, multiplier)
                buy_df,sell_df = backtest(df_temp.copy())
                buy_df_trailing = buy_df.copy()
                sell_df_trailing = sell_df.copy()
                
                ## check every take profit points for calculation. if data hits in the range it will return true for that value
                for percentage in percentage_list:
                    buy_df_trailing,sell_df_trailing = trailing_hit_check(df_temp, buy_df_trailing, sell_df_trailing, percentage)
                 
                    
                ## calculation of return for trailing take profit data
                total_return = 0
                total_return_list = pd.DataFrame(columns=[str(period) + '_' + str(multiplier)])

                for index, row in buy_df_trailing.iterrows():
                    counter = 0
                    if row[-5] == True:
                        counter = counter + 1
                        total_return = total_return + 0.2 * percentage_list[0]
                    if row[-4] == True:
                        counter = counter + 1
                        total_return = total_return + 0.2 * percentage_list[1]
                    if row[-3] == True:
                        counter = counter + 1
                        total_return = total_return + 0.2 * percentage_list[2]
                    if row[-2] == True:
                        counter = counter + 1
                        total_return = total_return + 0.2 * percentage_list[3]
                    if row[-1] == True:
                        counter = counter + 1
                        total_return = total_return + 0.2 * percentage_list[4]
                        
                    total_return = (1 - counter * 0.2) * row['Return'] * 100 + total_return
                    total_return_list.loc[len(total_return_list)] = (total_return/100)
                    
                return_df_buy_trailing_tp[str(period) + '_' + str(multiplier)] = total_return_list[str(period) + '_' + str(multiplier)]
                
                total_return = 0
                total_return_list = pd.DataFrame(columns=[str(period) + '_' + str(multiplier)])

                for index, row in sell_df_trailing.iterrows():
                    counter = 0
                    if row[-5] == True:
                        counter = counter + 1
                        total_return = total_return + 0.2 * percentage_list[0]
                    if row[-4] == True:
                        counter = counter + 1
                        total_return = total_return + 0.2 * percentage_list[1]
                    if row[-3] == True:
                        counter = counter + 1
                        total_return = total_return + 0.2 * percentage_list[2]
                    if row[-2] == True:
                        counter = counter + 1
                        total_return = total_return + 0.2 * percentage_list[3]
                    if row[-1] == True:
                        counter = counter + 1
                        total_return = total_return + 0.2 * percentage_list[4]
                        
                    total_return = (1 - counter * 0.2) * row['Return'] * 100 + total_return
                    total_return_list.loc[len(total_return_list)] = (total_return/100)
                    
                return_df_sell_trailing_tp[str(period) + '_' + str(multiplier)] = total_return_list[str(period) + '_' + str(multiplier)]
         
                return_df_buy[str(period) + '_' + str(multiplier)] = (buy_df['Return']).cumsum()
                return_df_sell[str(period) + '_' + str(multiplier)] = (buy_df['Return']).cumsum()
                
        
        ## write to csv.
        return_df_buy.to_csv('C:/Users/kisah/Desktop/Crypto_Data/Supertrend_results/buy_' + symbol + dataframe +'MINUTE2020-2023.csv')
        return_df_sell.to_csv('C:/Users/kisah/Desktop/Crypto_Data/Supertrend_results/sell_' + symbol + dataframe +'MINUTE2020-2023.csv')
        return_df_buy_trailing_tp.to_csv('C:/Users/kisah/Desktop/Crypto_Data/Supertrend_results/buy_trailingtp_' + symbol + dataframe +'MINUTE2020-2023.csv')
        return_df_sell_trailing_tp.to_csv('C:/Users/kisah/Desktop/Crypto_Data/Supertrend_results/sell_trailingtp_' + symbol + dataframe +'MINUTE2020-2023.csv')

