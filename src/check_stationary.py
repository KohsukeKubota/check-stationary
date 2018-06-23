import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def ADF_test(df, row_name):
    if df[row_name].isnull().any():
        df_test = adfuller(df[row_name].dropna(), autolag='AIC')
    else:
        df_test = adfuller(df[row_name], autolag='AIC')
    df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value',
                                               '#Lags Used', 'Number of Observations Used'])
    for k, v in df_test[4].items():
        df_output['Critical Value ({})'.format(k)] = v
    return df_output

if __name__ == '__main__':
    # データの読み込みとTimeIndexの設定
    df = pd.read_csv('../data/AirPassengers.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    df.columns = ['Passengers']

    # ADF検定（帰無仮説は「単位根が存在する」、対立仮説が「定常である」）
    # サンプルデータでは帰無仮説を棄却できない
    df_output = ADF_test(df, 'Passengers')
    print(df_output)

    # 差分系列に対してADF検定を行う(帰無仮説を棄却できれば、データ系列は単位根過程）
    diff_df = pd.DataFrame(index=df.index, columns=df.columns)
    diff_df['Passengers'] = df['Passengers'] - df['Passengers'].shift(1)
    diff_df = diff_df.dropna()
    df_output = ADF_test(diff_df, 'Passengers')
    print(df_output)
