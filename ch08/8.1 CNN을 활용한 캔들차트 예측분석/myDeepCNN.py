import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import math
import json
import sys

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.layers import Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import *
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from utils.dataset import dataset as dataset
from utils.dataset import dataset
import argparse

import time
from datetime import timedelta


def build_dataset(data_directory, img_width):
    # X, y, tags = dataset.dataset(data_directory, int(img_width))
    X, y, tags = dataset(data_directory, int(img_width))
    print(len(tags))
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count
    print("train size : {}".format(train_size))
    feature = X
    label = np_utils.to_categorical(y, nb_classes)
    return feature, label, nb_classes


# 모델 구조
def build_model(SHAPE, nb_classes, bn_axis, seed=None):

    if seed:
        # 예측 result를 재현하기 위해 특정 sedd 값을 설정
        np.random.seed(seed)
    # 입력값을 전달받는 층을 정의
    input_layer = Input(shape=SHAPE)

    # Step 1
    # 2차원 합성곱층(Conv2D)을 정의, input_layer를 함수형으로 연결
    x = Conv2D(32, 3, 3, init='glorot_uniform',
               border_mode='same', activation='relu')(input_layer)
    # Step 2 - Pooling
    # Max Pooling Layer를 정의, feature 값 추출을 위해 존재하는 층 (CNN 구성)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Step 1
    # Conv2D 추가
    x = Conv2D(48, 3, 3, init='glorot_uniform', border_mode='same',
               activation='relu')(x)
    # Step 2 - Pooling
    # MaxPooling2D 추가
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 신경망 학습시 과적합 방지를 위해 일부 연결층을 제거하는 drop out 레이어를 추가
    x = Dropout(0.25)(x)

    # Step 1
    x = Conv2D(64, 3, 3, init='glorot_uniform', border_mode='same',
               activation='relu')(x)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Step 1
    x = Conv2D(96, 3, 3, init='glorot_uniform', border_mode='same',
               activation='relu')(x)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Step 3 - Flattening
    x = Flatten()(x)

    # Step 4 - Full connection

    x = Dense(output_dim=256, activation='relu')(x)
    # Dropout
    x = Dropout(0.5)(x)

    x = Dense(output_dim=2, activation='softmax')(x)
    # 최종 연결된 출력층 (x)과 최초 입력층 (input_layer)을 전달해 모델 구축
    model = Model(input_layer, x)

    return model


def main():
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='an input directory of dataset', required=True)
    parser.add_argument('-d', '--dimension',
                        help='a image dimension', type=int, default=48)
    parser.add_argument('-c', '--channel',
                        help='a image channel', type=int, default=3)
    parser.add_argument('-e', '--epochs',
                        help='num of epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size',
                        help='num of batch_size', type=int, default=64)
    # parser.add_argument('-o', '--optimizer',
    #                     help='choose the optimizer (rmsprop, adagrad, adadelta, adam, adamax, nadam)', default="adam")
    parser.add_argument('-o', '--output',
                        help='a result file', type=str, default="hasilnya.txt")
    args = parser.parse_args()
    # dimensions of our images.
    img_width, img_height = args.dimension, args.dimension
    channel = args.channel
    epochs = args.epochs
    batch_size = args.batch_size
    SHAPE = (img_width, img_height, channel)
    bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1

    # channel = 4
    # bn_axis = 4 if K.image_dim_ordering() == 'tf' else 1

    data_directory = args.input

    print("loading dataset")
    X_train, Y_train, nb_classes = build_dataset(
        "{}/train".format(data_directory), args.dimension)
    X_test, Y_test, nb_classes = build_dataset(
        "{}/test".format(data_directory), args.dimension)
    print("number of classes : {}".format(nb_classes))

    # 모델 생성
    model = build_model(SHAPE, nb_classes, bn_axis)
    # 모델 학습
    # Adam optimizer를 선언
    # 범주형 엔트로피 손실 함수 설정
    # 학습 과정에서 정확도를 관찰한다
    model.compile(optimizer=Adam(lr=1.0e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    # 검증 데이터 없이 오로지 학습 데이터로만 학습 수행
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

    # Save Model or creates a HDF5 file
    # 학습 완료된 모델을 재사용하기 위해 저장
    model.save(f'{epochs}epochs_{batch_size}batch_cnn_model_{data_directory}.h5'.replace("/", "_"), overwrite=True)
    # del model  # deletes the existing model
    # 테스트 데이터와 학습한 모델을 사용해 예측값 출력
    predicted = model.predict(X_test)
    # one-hot encoding으로 되어 있는 예측값과 실제 레이블 데이터를 다른 범주롤 변화한다
    y_pred = np.argmax(predicted, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    # sklearn에서 제공하는 혼동 행렬 분석 함수 호출
    cm = confusion_matrix(Y_test, y_pred)
    # 혼동 행렬에서 계산된 수치로 더 많은 수치를 계산하는 report 함수 호출
    report = classification_report(Y_test, y_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    if tp == 0:
        tp = 1
    if tn == 0:
        tn = 1
    if fp == 0:
        fp = 1
    if fn == 0:
        fn = 1
    TPR = float(tp)/(float(tp)+float(fn))
    FPR = float(fp)/(float(fp)+float(tn))
    accuracy = round((float(tp) + float(tn))/(float(tp) +
                                              float(fp) + float(fn) + float(tn)), 3)
    specitivity = round(float(tn)/(float(tn) + float(fp)), 3)
    sensitivity = round(float(tp)/(float(tp) + float(fn)), 3)
    mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/math.sqrt(
        (float(tp)+float(fp))
        * (float(tp)+float(fn))
        * (float(tn)+float(fp))
        * (float(tn)+float(fn))
    ), 3)

    f_output = open(args.output, 'a')
    f_output.write('=======\n')
    f_output.write('{}epochs_{}batch_cnn\n'.format(
        epochs, batch_size))
    f_output.write('TN: {}\n'.format(tn))
    f_output.write('FN: {}\n'.format(fn))
    f_output.write('TP: {}\n'.format(tp))
    f_output.write('FP: {}\n'.format(fp))
    f_output.write('TPR: {}\n'.format(TPR))
    f_output.write('FPR: {}\n'.format(FPR))
    f_output.write('accuracy: {}\n'.format(accuracy))
    f_output.write('specitivity: {}\n'.format(specitivity))
    f_output.write("sensitivity : {}\n".format(sensitivity))
    f_output.write("mcc : {}\n".format(mcc))
    f_output.write("{}".format(report))
    f_output.write('=======\n')
    f_output.close()
    end_time = time.monotonic()
    print("Duration : {}".format(timedelta(seconds=end_time - start_time)))

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # Plot a confusion matrix.
    # cm is the confusion matrix, names are the names of the classes.
    def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(names))
        plt.xticks(tick_marks, names, rotation=45)
        plt.yticks(tick_marks, names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Plot an ROC. pred - the predictions, y - the expected output.
    def plot_roc(pred, y):
        fpr, tpr, _ = roc_curve(y, pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig('ROC AUC.png')
        plt.show()


    plot_roc(y_pred, Y_test)


if __name__ == "__main__":
    main()
