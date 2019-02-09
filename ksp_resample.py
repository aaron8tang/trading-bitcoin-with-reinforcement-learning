import os.path as path
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot

script_path = path.dirname(__file__)


def OHLCV(df):
    keys = [
        'date',
        'o', 'h', 'l', 'c', 'v'
    ]

    tracker = {}
    for k in keys: tracker[k] = []

    groups = df.groupby(pd.TimeGrouper(freq='{0}Min'.format(15)))
    for group in groups:
        g1 = group[1]
        if len(g1) == 0: continue

        dt = group[0]
        tracker['dt'].append(dt)

        # extract OHLCV from bar data
        open = g1.ix[0]['Open']
        high = g1['High'].max()
        low = g1['Low'].min()
        close = g1.ix[-1]['Close']
        vol = g1['Volume'].sum().round(2)

        tracker['open'].append(open)
        tracker['high'].append(high)
        tracker['low'].append(low)
        tracker['close'].append(close)
        tracker['vol'].append(vol)

    df = pd.DataFrame(data=tracker, columns=keys[1:], index=tracker['dt'])
    return df


def feat_extract(df):
    import talib

    close = df['c'].values.astype(np.float)
    vol = df['v'].values.astype(np.float)
    df_ = pd.DataFrame(index=df.index)

    df_['r'] = np.log(talib.ROCR(close, timeperiod=1))
    df_['r_1'] = np.log(talib.ROCR(close, timeperiod=2))
    df_['r_2'] = np.log(talib.ROCR(close, timeperiod=3))

    r = df_['r'].values

    zscore = lambda x, timeperiod: (x - talib.MA(x, timeperiod)) / (talib.STDDEV(x, timeperiod) + 1e-8)
    df_['rZ12'] = zscore(r, 12)
    df_['rZ96'] = zscore(r, 96)

    change = lambda x, timeperiod: x / talib.MA(x, timeperiod) - 1
    df_['pma12'] = zscore(change(close, 12), 96)
    df_['pma96'] = zscore(change(close, 96), 96)
    df_['pma672'] = zscore(change(close, 672), 96)

    ma_r = lambda x, tp1, tp2: talib.MA(x, tp1) / talib.MA(x, tp2) - 1
    df_['ma4/36'] = zscore(ma_r(close, 4, 36), 96)
    df_['ma12/96'] = zscore(ma_r(close, 12, 96), 96)

    def acc(x, tp1, tp2):
        x_over_avg = x / talib.MA(x, tp1)
        value = x_over_avg / talib.MA(x_over_avg, tp2)
        return value

    df_['ac12/12'] = zscore(acc(close, 12, 12), 96)
    df_['ac96/96'] = zscore(acc(close, 96, 12), 96)

    df_['vZ12'] = zscore(vol, 12)
    df_['vZ96'] = zscore(vol, 96)
    df_['vZ672'] = zscore(vol, 672)

    df_['vma12'] = zscore(change(vol, 12), 96)
    df_['vma96'] = zscore(change(vol, 96), 96)
    df_['vma672'] = zscore(change(vol, 672), 96)

    df_['vol12'] = zscore(talib.STDDEV(r, 12), 96)
    df_['vol96'] = zscore(talib.STDDEV(r, 96), 96)
    df_['vol672'] = zscore(talib.STDDEV(r, 672), 96)

    df_['dv12/96'] = zscore(change(talib.STDDEV(r, 12), 96), 96)
    df_['dv96/672'] = zscore(change(talib.STDDEV(r, 96), 672), 96)

    df_ = df_.fillna(0.)
    assert (not df.isnull().values.any()), 'feature dframe contain NaNs'
    return df_


def plot_series(s):
	s.plot()
	pyplot.show()


if __name__ == '__main__':
    
    # load csv
    save_path = path.join(script_path, 'data.csv')
    df = pd.read_csv(save_path,
                     index_col=[0],
                     parse_dates=True)

    # convert unix timestamp to datetime
    # df.index = pd.to_datetime(df.index, unit='s')
    df.index = pd.to_datetime(df.index)

    # select period between Dec. 1, 2014 ~ Jun. 14, 2017
    '''
    start_date = pd.Timestamp(year=2014, month=12, day=1, hour=0, minute=0)
    end_date = pd.Timestamp(year=2017, month=6, day=14, hour=23, minute=59)
    mask = (df.index >= start_date) & (df.index <= end_date)
    df = df.loc[mask]
    '''

    # drop and change column names
    df = df[['o', 'h', 'l', 'c', 'v']]
    # df = OHLCV(df)

    # # save the csv file for further use if you want
    # save_path = path.join(script_path, 'BTCUSD-15Min.csv')
    # df.to_csv(save_path)
    # df = pd.read_csv(save_path,
    #                  index_col=[0],
    #                  parse_dates=True)

    feat_df = feat_extract(df)

    data_dict = {
        'data': feat_df,
        'label': df
    }

    plot_series(df['c'])

    save_path = path.join(script_path, 'data.pkl')
    with open(save_path, mode='wb') as handler:
        pickle.dump(data_dict, handler, protocol=pickle.HIGHEST_PROTOCOL)
