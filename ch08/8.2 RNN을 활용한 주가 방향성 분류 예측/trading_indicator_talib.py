# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib

# import scipy.stats as stats
# from sklearn.preprocessing import MinMaxScaler

# 데이터불러오기
df = pd.read_csv('../../data/ch08/intc.csv',
                 index_col='Date',
                 parse_dates=True)
sox_df = pd.read_csv('../../data/ch08/sox_df.csv',
                     index_col='Date',
                     parse_dates=True)
vix_df = pd.read_csv('../../data/ch08/vix_df.csv',
                     index_col='Date', parse_dates=True)
snp500_df = pd.read_csv('../../data/ch08/s&p500.csv',
                        index_col='Date',
                        parse_dates=True)

# 데이터불러오기
df['next_rtn'] = df['Close'] / df['Open'] - 1  # 1
df['log_return'] = np.log(1 + df['AdjClose'].pct_change())  # 1
# 이동평균(MovingAverage)
df['MA5'] = talib.SMA(df['Close'], timeperiod=5)  # 1
df['MA10'] = talib.SMA(df['Close'], timeperiod=10)
df['RASD5'] = talib.SMA(talib.STDDEV(df['Close'], timeperiod=5, nbdev=1), timeperiod=5)  # 1
df['RASD10'] = talib.SMA(talib.STDDEV(df['Close'], timeperiod=5, nbdev=1), timeperiod=10)

# MACD(MovingAverageConvergence&Divergence)지표
macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26,
                                        signalperiod=9)
df['MACD'] = macd  # 1

# 모멘텀지표
# CCI:CommodityChannelIndex
df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)  # 1

# 변동성지표
# #ATR: Average True Range
df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)  # 1

# 볼린저밴드
upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
df['ub'] = upper  # 1
df['middle'] = middle
df['lb'] = lower  # 1

# MTM1 MTM3
df['MTM1'] = talib.MOM(df['Close'], timeperiod=1)  # 1
df['MTM3'] = talib.MOM(df['Close'], timeperiod=3)

# Rate of Change지표
df['ROC'] = talib.ROC(df['Close'], timeperiod=60)  # 1
# Williams' %R
df['WPR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)  # 1

# 가공된 데이터 프레임 옆에 시장 지수 데이터 추가
snp500_df = snp500_df.loc[:, ['Close']].copy()
snp500_df.rename(columns={'Close': 'S&P500'}, inplace=True)  # 1
sox_df = sox_df.loc[:, ['Close']].copy()
sox_df.rename(columns={'Close': 'SOX'}, inplace=True)
vix_df = vix_df.loc[:, ['Close']].copy()
vix_df.rename(columns={'Close': 'VIX'}, inplace=True)
df = df.join(snp500_df, how='left')  # 1
df = df.join(sox_df, how='left')
df = df.join(vix_df, how='left')

# result
df.head()
df.columns

# 특성목록
feature1_list = ['Open', 'High', 'Low', 'AdjClose', 'Volume', 'log_return']
feature2_list = ['RASD5', 'RASD10', 'ub', 'lb', 'CCI', 'ATR', 'MACD', 'MA5', 'MA10', 'MTM1', 'MTM3', 'ROC', 'WPR']
feature3_list = ['S&P500', 'SOX', 'VIX']
feature4_list = ['next_rtn']
all_features = feature1_list + feature2_list + feature3_list + feature4_list
phase_flag = '3'

if phase_flag == '1':
    train_from = '2010-01-04'
    train_to = '2012-01-01'

    val_from = '2012-01-01'
    val_to = '2012-04-01'

    test_from = '2012-04-01'
    test_to = '2012-07-01'

elif phase_flag == '2':
    train_from = '2012-07-01'
    train_to = '2014-07-01'

    val_from = '2014-07-01'
    val_to = '2014-10-01'

    test_from = '2014-10-01'
    test_to = '2015-01-01'

else:
    train_from = '2015-01-01'
    train_to = '2017-01-01'

    val_from = '2017-01-01'
    val_to = '2017-04-01'

    test_from = '2017-04-01'
    test_to = '2017-07-01'

# 학습/검증/테스트
train_df = df.loc[train_from:train_to, all_features].copy()  # 1
val_df = df.loc[val_from:val_to, all_features].copy()  # 1
test_df = df.loc[test_from:test_to, all_features].copy()  # 1
