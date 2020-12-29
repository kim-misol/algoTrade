from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

from .trading_indicator_talib import *

# LSTM 모델 생성
K.clear_session()
# 입력 데이터셋 형태에 맞게 값을 지정한다.
# 데이터 구조가 3차원으로 구성되었기 때문에 [None, x_train.shape[1], x_train.shape[2]]으로 값을 전달
# 케라스에서 첫 번째 차원에는 데이터의 개수가 들어가는데, 모든 임의의 스칼라 any scalar 를 의미하는 None을 넣어준다.
# 데이터 양이 많아지면 None으로 대체할 수 있다.
# 두 번째 차원은 입력 데이터의 시간 축을 의미하고,
# 세 번째 축은 LSTM 입력층에 한 번에 입력되는 데이터 개수를 나타낸다. 즉 특성 feature 데이터(설명 변수 데이터)라고 보면 된다
input_layer = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))
# 다층 구조로 구성된 LSTM이다. (LSTM 층 위에 LSTM 층이 연결)
# 이전 층에서 사용한 출력이 다음 층에 전달되어야 하기 때문에 return_sequences = True을 사용
# 마지막 층에서는 옵션을 제외한다.
# 또한 functional 방법에서는 각 클래스 내부에 callback() 함수가 정의되어 있어,
# LSTM(input)(input_layer)와 같이 이전 층을 매개변수로 전달하면서 층을 이어준다.
layer_lstm_1 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(input_layer)
# 배치정규화층을 이어준다.
# 배치정규화는 신경망의 입력값을 평균 0 분산 1로 정규화 하여 네트워크 학습이 잘 일어나도록 돕는 방식
layer_lstm_1 = BatchNormalization()(layer_lstm_1)
# LSTM 각 층에 L2 규제를 적용하면서 계속해서 층을 이어간다.
layer_lstm_2 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm_1)
# 드롭아웃층을 이어 임의의 확률로 가중치 선을 지운다.
# 네트워크 학습 시 과적함 방지
layer_lstm_2 = Dropout(0.25)(layer_lstm_2)
# LSTM층 → BatchNormalize → LSTM → Dropout 을 반복하면서 층을 쌓아간다.
layer_lstm_3 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm_2)
layer_lstm_3 = BatchNormalization()(layer_lstm_3)
layer_lstm_4 = LSTM(num_unit, return_sequences=True, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm_3)
layer_lstm_4 = Dropout(0.25)(layer_lstm_4)
layer_lstm_5 = LSTM(num_unit, recurrent_regularizer=regularizers.l2(0.01))(layer_lstm_4)
layer_lstm_5 = BatchNormalization()(layer_lstm_5)
# 완전 연결층으로 연결되면서 최종 예측값 출력
# 완전 연결층 (fully connected layer) Dense
output_layer = Dense(2, activation='sigmoid')(layer_lstm_5)
# 입력층과 출력층을 연결해 모델 객체를 만들어낸다.
model = Model(input_layer, output_layer)
# 모델 학습 방식을 설정해 모델을 결정한다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
# x_train, y_train 훈련 데이터셋으로 모델 학습
# epochs 반복 훈련 횟수, 시간이 많이 걸려 편의상 20으로 설정 (논문에서는 5000)
# batch_size를 설정하고 unrolled cells를 이곳에 지정
"""
LSTM은 RNN의 문제점을 해결하기 위해 고안된 신경망 모델이다. 
RNN에서 재귀하는 과정을 펼쳐서 시각화한 것을 ‘Unrolled’라고 하며, Unrolled 되었을 때 펼쳐진 블록들을 cell이 라고 해석한다. 
결국 펼쳐진 셀(unrolled cells )이라는 것은 결국 몇 번 재귀순환을 할 것인지를 설명하게 된다.
펼침 과정을 LSTM에서 실행하기 위해 return_sequences=True 속성을 부여했다. 
"""
# verbose 옵션에 따라 학습 진행 중 로그를 어떻게 볼지 설정 (0=silent, 1=progress bar, 2=one line per epoch)
# validation_data를  통해 epoch가 끝날 때마다 학습 데이터로 평가
history = model.fit(x_train, y_train, epochs=20, batch_size=10, verbose=1, validation_data=(x_val, y_val))