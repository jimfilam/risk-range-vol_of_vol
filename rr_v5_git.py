# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import scipy.special as scsp
from pandas.tseries.offsets import BDay # use to find index
import warnings
from datetime import datetime, timedelta, date
from hurst import compute_Hc
import pandas_market_calendars as mcal
import config
import get_data
import logging
# import matplotlib.pyplot as plt
from tda import auth, client

logging.basicConfig(filename='logging.txt', level=logging.ERROR)

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""### Rough Vol Functions
this functions from the website link
https://tpq.io/p/rough_volatility_with_python.html
"""

def test_hurst(df_h):

    # np.cumsum(random_increments)  # create a random walk from random
    # increments
    series = df_h

    # Evaluate Hurst equation
    H, c, data = compute_Hc(series, kind='price')
    #     assert H<0.6 and H>0.4
    return H, c, data


def call_hurst(df, column, y):
    # Hurst Numbers
    df = df.reset_index()
    df_h = df[column]
    h_array = np.array([])
    c_array = np.array([])
    h_counter = np.array([])

    # y = increment for hurst 391 = full market day
    for x in range(0, len(df_h), y):
        h, c, data = test_hurst(df_h[x:x + y])
        h_array = np.append(h_array, h)
        c_array = np.append(c_array, c)
        h_counter = np.append(h_counter, x)
    
    # h_count = h_counter/y
    # print ( '\n Hurst Data  \n' , (h_array) )
    # print ( '\n Number of Days or Groups = %.0f days' %(h_count[-1] ) )
    # print ( '\n Hurst Mean = %.2f ' %( h_array.mean() ) )
    # print ( ' Hurst Last 5 = %.2f ' %( h_array[-5:].mean() ) )
    return h_array, c_array


def dlsig2(sig, x, pr=False):
  if pr:
      a= np.array([(sig-sig.shift(lag)).dropna() for lag in x])
      a=a ** 2
      print (a.info() )
  return [np.mean((sig-sig.shift(lag)).dropna() ** 2) for lag in x]

def c_tilde(h):
    return scsp.gamma(3. / 2. - h) / scsp.gamma(h + 1. / 2.) \
        * scsp.gamma(2. - 2. * h)

def forecast_XTS(rvdata, h, date, nLags, delta, nu):
    i = np.arange(nLags)
    cf = 1. / ((i + 1. / 2.) ** (h + 1. / 2.) * (i + 1. / 2. + delta))
    ldata = rvdata.truncate(after=date)
    lenth_data = len(ldata)
    ldata = np.log(ldata.iloc[lenth_data - nLags:])
    ldata['cf'] = np.fliplr([cf])[0]
    ldata = ldata.dropna()
    fcst = (ldata.iloc[:, 0] * ldata['cf']).sum() / sum(ldata['cf'])
    return math.exp(fcst + 2 * nu ** 2 * c_tilde(h) * delta ** (2 * h))

def rv_vol_vol(var, OxfordH):
    window = 2
    rvdata = pd.DataFrame(var)
    nu = OxfordH['nu_est'][0]  # Vol of vol estimate
    h = OxfordH['h_est'][0]
    n = len(var)
    delta = 1
    nLags = window  # in exaample this was 500
    dates = rvdata.iloc[nLags:n - delta].index
    rv_pred = [forecast_XTS(rvdata, h=h, date=d, nLags=nLags,
                            delta=delta, nu=nu) for d in dates]
    rv_act = rvdata.iloc[nLags + delta:n].values
    return rv_pred, rv_act

def var_func(df):
    ''' This uses minute data to get the variance for the day 
    still not quite right when compared to SPY Oxford Data
    https://realized.oxford-man.ox.ac.uk/ script from 
    https://github.com/BayerSe/RealizedQuantities '''

    def realized_quantity(fun):
        """Applies the function 'fun' to each day separately"""
        return intraday_returns.groupby(pd.Grouper(freq="B")).apply(fun)[index]

    # Calculation HF variance by Sheppard and basic realized var.
    # df['date'] = df.timestamp.dt.tz_convert(None)
    df.set_index('datetime', inplace = True)
    data = df['close']
    intraday_returns = data.groupby(pd.Grouper(freq="B")).apply(lambda x: np.log(x / x.shift(1))).dropna()
    # print ('intraday ' , intraday_returns )
    intraday_returns.dropna(inplace=True)
    index = data.groupby(pd.Grouper(freq="B")).first().dropna().index

    ''' Sherpherd var. '''
    mu_1 = np.sqrt(( 2 / np.pi ))
    # var = mu_1 ** (-2) * realized_quantity(lambda x: (x.abs() * x.shift(1).abs()).sum())

    ''' Realized var '''
    var = realized_quantity(lambda x: (x ** 2).sum())
    print ('Using HIGH RATE VAR calculations')
    return var

def var_minute (dfmt, window):
    ''' This is to get Variance using Minute Data Stream 
    and works with daily data '''
    # self._df.set_index('datetime', inplace=True)
    data = dfmt.close
    index = data.groupby(pd.Grouper(freq="B")).first().dropna().index
    dfd = dfmt.groupby(pd.Grouper(freq='B') ).last().dropna().copy()
    dfd['r_log'] =np.log(dfd.close/dfd.close.shift(1)).dropna()
    var_by_day = dfd['r_log'].rolling(window=window).var().dropna()
    print ('Using SIMPLE VAR calculations')
    return var_by_day

def get_time_peroid(days):
    now = pd.Timestamp.now()
    end_dt = now - pd.Timedelta(now.strftime('%H:%M:%S')
                                ) + pd.Timedelta('1 day')
    start_dt = end_dt - pd.Timedelta( str(days) + ' days') # str(start_days)
    _from = start_dt.strftime('%Y-%m-%d')
    _to = end_dt.strftime('%Y-%m-%d')
    return _from, _to

def m_time(df, ndays):
    """takes stock df from TD and ensure 1 min market time is used 
    only for open market"""
    # print (self.dfa)
    # dfa.set_index('datetime', inplace=True)
    df_t = df.resample('T').ffill().bfill().reset_index()
    # _from = '2021-01-15'
    # to = '2021-03-05'
    _from, to = get_time_peroid(ndays+5)
    nyse = mcal.get_calendar('NYSE')
    early = nyse.schedule(start_date=_from, end_date=to)
    dfm = pd.DataFrame()
    df_t['dat'] = pd.to_datetime(df_t.datetime, utc=True)
    for i in range(0, len(early)):
        df2 = df_t[(df_t.dat >= early.market_open[i]) &
                (df_t.dat <= early.market_close[i])]
        dfm = dfm.append(df2)
    dfm.drop(columns=['dat'], inplace=True)
    dfm.set_index('datetime', inplace=True)
    return dfm


def rr_calc(symbols, window, ndays):
    message = []
    rr = pd.DataFrame(columns=['date','symbol', 'lower', 'upper', 'close','Low %',
                                'High %','Range','max high','min low', 'Predict', 'Actual', 
                                'hurst' ,'nu','h_var'] )
    for symbol in symbols:
        symbol = symbol.upper()
        try:
            ''' get stock OHLC data call.
            return data in a pandas dataframe '''
            dmt = get_data.td_historical_min(symbol)

            var = var_func(dmt)
            """ var_minute was giving SVD fit error
                so using var_func work some times"""
            # var = var_minute(dmt, window)
        
            x = np.arange(1, len(var))
            h = list()
            nu = list() # nu is vol_of_vol
            sig = np.log(np.sqrt(var)).dropna()  # sig is std
            
            model = np.polyfit(np.log(x), np.log(dlsig2(sig, x)), 2)
            nu.append(np.sqrt(np.exp(model[1]))) 
            h.append(model[0] / 2.)  # Don't have a clue why hurst std is diveded by 2
            OxfordH = pd.DataFrame({'h_est': h, 'nu_est': nu})
        
            rv_predict, rv_actual = rv_vol_vol(var, OxfordH)  # passes nu and h in Oxford df
            vol_actual = np.sqrt(np.multiply(rv_actual, 252))
            vol_predict = np.sqrt(np.multiply(rv_predict, 252))

            # Risk Ranges Using expon rolling mean an nu which labeled as vol_of_vol
            df_max = dmt.high.groupby(pd.Grouper(freq="B")).max().dropna()
            df_min = dmt.low.groupby(pd.Grouper(freq="B")).min().dropna()
            df_close = dmt.close.groupby(pd.Grouper(freq="B")).last().dropna()
            # df_ema_min = df_min.ewm(com=1 ).mean()
            # df_ema_max = df_max.ewm(com=1 ).mean()

            df_ema_min = df_min.ewm(span=window, min_periods=0,adjust=False,ignore_na=False).mean()
            df_ema_max = df_max.ewm(span=window, min_periods=0,adjust=False,ignore_na=False).mean()
           

            H_C = abs(df_ema_max - df_close)
            L_C = abs(df_ema_min - df_close)
            #############################################################

            """ Risk Ranges Using H_C and L_C using nu """
            
            x = H_C * nu[0] + H_C  
            RR_upper = df_close + x

            y = L_C * nu[0] + L_C
            RR_lower = df_close - y

            """Risk Ranges Using H_C and L_C using Predicted vol """
            # x = H_C * vol_actual[-1] + H_C  
            # RR_upper = df_close + x

            # y = L_C * vol_actual[-1]  + L_C
            # RR_lower = df_close - y

            #############################################################
            last_price = df_close[-1:].values
            max_high = df_close.max()
            min_low = df_close.min()

            ############## Hurst #########################################
            """ Using Hurst function """
            hurst, c_array = call_hurst(dmt, 'close', 391) 
            # print (hurst, c_array)

            dic_new = {'date':df_close.index[-1:][0], 
                        'symbol':symbol,
                        'lower':RR_lower[-1:][0], 
                        'upper':RR_upper[-1:][0], 
                        'close':last_price[0], 
                        'max high':max_high,
                        'min low':min_low, 'nu':nu[0], 
                        'h_var': h[0],
                        'Predict':vol_predict[-1:][0] * 100,
                        'Actual':vol_actual[-1:][0][0] * 100, 
                        'Range': RR_upper[-1:][0] - RR_lower[-1:][0],
                        'hurst':hurst[-1:][0]                       
                        }

            rr = rr.append( dic_new, ignore_index=True )
        except Exception as e:
            vol_predict = 0
            vol_actual = 0
            print (e)
            logging.error(symbol, exc_info=True)
            message.append(symbol)

    return rr, vol_actual, vol_predict, var, h, dmt

def rr(symbols, window, ndays):
    rr, vol_actual, vol_predict, var, h, dmt = rr_calc(symbols, 
                                                        window, 
                                                        ndays)  
                                                                
    rr['Low %']= (1 - rr.lower / rr.close ) * 100
    rr['High %'] = (rr.upper / rr.close - 1) * 100
    rr['buy'] = rr.Range * .1 + rr.lower
    rr['pct_ratio'] = rr['Low %']/ rr['High %']
    rr['pct_diff'] = (rr['High %'] - rr['Low %']) / rr['High %']
    rr.sort_values('pct_diff', ascending=False, inplace=True)
    return rr

''' window: is for the Exp moving average rolling
in pandas'''
window = 2

''' ndays: used only in calendar (def m-time). If you get 10 days of data
    then I add 5 days to ensure enough open makert days'''
ndays = 10 


symbols = ['xle', 'a']

rr = rr(symbols, window, ndays)
print (rr)






