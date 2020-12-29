# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib

# import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler

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
# 다음날 수익률 예측하도록
df['next_rtn'] = df['Close'] / df['Open'] - 1
df['log_return'] = np.log(1 + df['Adj Close'].pct_change())
# 이동평균(MovingAverage)
df['MA5'] = talib.SMA(df['Close'], timeperiod=5)
df['MA10'] = talib.SMA(df['Close'], timeperiod=10)
df['RASD5'] = talib.SMA(talib.STDDEV(df['Close'], timeperiod=5, nbdev=1), timeperiod=5)
df['RASD10'] = talib.SMA(talib.STDDEV(df['Close'], timeperiod=5, nbdev=1), timeperiod=10)

# MACD(MovingAverageConvergence&Divergence)지표
macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26,
                                        signalperiod=9)
df['MACD'] = macd

# 모멘텀지표
# CCI:CommodityChannelIndex
df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)

# 변동성지표
# #ATR: Average True Range
df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

# 볼린저밴드
upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
df['ub'] = upper
df['middle'] = middle
df['lb'] = lower

# MTM1 MTM3
df['MTM1'] = talib.MOM(df['Close'], timeperiod=1)
df['MTM3'] = talib.MOM(df['Close'], timeperiod=3)

# Rate of Change지표
df['ROC'] = talib.ROC(df['Close'], timeperiod=60)
# Williams' %R
df['WPR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

# 가공된 데이터 프레임 옆에 시장 지수 데이터 추가
snp500_df = snp500_df.loc[:, ['Close']].copy()
# 중복 컬럼 간 구분을 위해서 Close 컬럼에 저장된 값을 종목명으로 변환
snp500_df.rename(columns={'Close': 'S&P500'}, inplace=True)
sox_df = sox_df.loc[:, ['Close']].copy()
sox_df.rename(columns={'Close': 'SOX'}, inplace=True)
vix_df = vix_df.loc[:, ['Close']].copy()
vix_df.rename(columns={'Close': 'VIX'}, inplace=True)
# 가공한 데이터프레임에 left join
df = df.join(snp500_df, how='left')
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


# 최소 최대 정규화
def min_max_normal(tmp_df):
    eng_list = []
    sample_df = tmp_df.copy()
    for x in all_features:
        if x in feature4_list:
            continue
        series = sample_df[x].copy()
        values = series.values
        values = values.reshape((len(values), 1))
        # 스케일러 생성 및 훈련
        # sklearn 라이브러리에서 정규화 객체를 받는다.
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 입력 데이터에 대해 정규화 범위 탐색
        scaler = scaler.fit(values)
        # 데이터셋 정규화 및 출력
        # 입력 데이터를 최소 최대 정규화
        normalized = scaler.transform(values)
        new_feature = '{}_normal'.format(x)
        eng_list.append(new_feature)
        # 정규화된 데이터를 새로운 컬럼명으로 저장
        sample_df[new_feature] = normalized
    return sample_df, eng_list


train_sample_df, eng_list = min_max_normal(train_df)
val_sample_df, eng_list = min_max_normal(val_df)
test_sample_df, eng_list = min_max_normal(test_df)


# 다음 영업일 종가 예측
# 특성 데이터롸 레이블 데이터 구분
def create_dateset_binary(data, feature_list, step, n):
    # lstm에 넣어줄 변수 데이터 설정
    train_xdata = np.array(data[feature_list[0:n]])
    # 마지막 단계 설정
    m = np.arange(len(train_xdata) - step)
    x, y = [], []
    for i in m:
        # 학습 데이터 기간 설정 (얼마만큼의 과거 데이터 기간을 전달할지)
        a = train_xdata[i:(i + step)]

        x.append(a)
    # 신경망 학습에 사용할 수 있게 데이터 정리 (3차원 형태)
    # len(m)은 한 칸씩 미뤘을 때 회대한 만들 수 있는 사각형의 개수 (batch_size)
    x_batch = np.reshape(np.array(x), (len(m), step, n))
    # 레이블링 데이터 (레이블 데이터: 다음날 종가)
    train_ydata = np.array(data[[feature_list[n]]])
    # n_step 이상부터 답을 사용할 수 있다.
    for i in m + step:
        # 시작 종가 설정 (이진 분류를 위해)
        start_price = train_ydata[i - 1][0]
        # 종료 종가 설정
        end_price = train_ydata[i][0]
        # 종료 종가가 크면 다음날 오를 것이라는 뜻으로 방향성을 레이블로 설정
        if end_price > start_price:
            label = 1  # 오름
        else:
            label = 0  # 내림
        # 임시로 생성된 레이블을 순차적으로 저장
        y.append(label)
    # 학습을 위한 1차원 열 벡터 형태로 변형
    y_batch = np.reshape(np.array(y), (-1, 1))
    return x_batch, y_batch


# 훈력, 검증, 테스트 데이터에 대해 레이블링 데이터 나누기
num_step = 5
num_unit = 200
n_feature = len(eng_list) - 1
# 훈련 데이터에 대한 변수 데이터와 레이블 데이터를 나눔
x_train, y_train = create_dateset_binary(train_sample_df[eng_list], eng_list, num_step, n_feature)
# 검증 데이터에 대한 변수 데이터와 레이블 데이터를 나눔
x_val, y_val = create_dateset_binary(val_sample_df[eng_list], eng_list, num_step, n_feature)
# 테스트 데이터에 대한 변수 데이터와 레이블 데이터를 나눔
x_test, y_test = create_dateset_binary(test_sample_df[eng_list], eng_list,  num_step, n_feature)

# 데이터 분리 함수 실행 후 반환된 입력 데이터 구조
x_train.shape   # (497, 5, 22) 22개의 변수가 있는 5일치 데이처로 학습, 총 497 묶음
