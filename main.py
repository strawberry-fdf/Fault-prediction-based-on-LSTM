import time
import warnings
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import newaxis
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM
from keras.layers.core import Dense, Activation, Dropout

path = './data.csv'

df = pd.read_csv('./data.csv')


# print(df.shape)


def load_data(filename, seq_len, normalise_window):
    data = open(filename, "rb").readlines()
    result = []

    sequence_length = seq_len + 2

    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    # print(data)
    re = np.array(result).astype('float32')
    ro = int(round(0.9 * len(result)))

    # window是最终的测试结果的真实值
    window = re[ro:, :1]
    if normalise_window:
        result = normalise_windows(result)
    result = np.array(result)
    # 划分train、test
    row = round(0.9 * result.shape[0])
    row = int(row)

    train = result[:row, :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1:]
    x_test = result[row:, :-1]
    y_test = result[row:, -1:]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return [x_train, y_train, x_test, y_test, window]


# 将输入结果进行归一化
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:  # window shape (sequence_length L ,)  即(51L,)
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


# 将预测的结果进行反归一化，获得真实预测结果
def FNormalise_windows(window, data):
    normalised_data = []
    for i in range(len(window)):  # window shape (sequence_length L ,)  即(51L,)
        normalised_data.append((float(data[i]) + 1) * float(window[i]))
    return normalised_data


def predict_all(model, data):
    predicted = model.predict(data)
    print('predicted_shape', np.array(predicted).shape)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


x_train, y_train, x_test, y_test, window = load_data(path, 50, True)

print('X_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# 创建神经网络

# 其中input_shape = (None,1)中的None代表模型输入集的数据量不限制
# units = 50 代表 代表将输入的维度映射成50个维度输出
# return_sequences为True意味着返回多个单元短期的输出结果,为False则只返回一个单元的输出结果
model = Sequential()
model.add(LSTM(input_shape=(None, 1), units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(input_shape=(None, 50), units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(input_shape=(None, 50), units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.add(Activation("linear"))

start = time.time()
model.compile(loss="mse", optimizer="rmsprop")
print("Compilation Time : ", time.time() - start)
