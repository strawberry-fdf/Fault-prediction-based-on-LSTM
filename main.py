import os
import numpy as np
import time
import pandas as pd
import keras
from matplotlib import pyplot
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM
from keras.layers.core import Dense, Activation, Dropout

'''
    对labels进行one-hot编码
'''


def label2hot(labels):
    values = np.array(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = values.reshape(len(values), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


'''
    处理训练数据
'''


def load_data(path1, path2, label1, label2):
    result = []
    files1 = os.listdir(path1)
    files2 = os.listdir(path2)

    for f1 in files1:
        data = pd.read_csv(path1 + '/' + f1)
        # 标签先给每一个戳都打上，最后在降维，降维成每个段一个标签
        data['labels'] = label1
        if 'TimeS' in data:
            data = data.drop(['TimeS'], axis=1)
        result.append(data)

    for f2 in files2:
        data = pd.read_csv(path2 + '/' + f2)
        data['labels'] = label2
        if 'TimeS' in data:
            data = data.drop(['TimeS'], axis=1)
        result.append(data)

    result = np.array(result)
    # 打乱顺序
    np.random.shuffle(result)
    row = round(0.8 * result.shape[0])
    row = int(row)
    # 通过均值降维
    label = result[:, :, -1:]
    label = label[:, :, :].mean(axis=2)
    label = label[:, :].mean(axis=1)
    label = label2hot(label)
    train = result[0:row, ::]

    x_train = train[:, :, :-1]
    y_train = label[0:row, :]

    x_test = result[row:, :, :-1]
    y_test = label[row:, :]

    print("Data processing is finished!")
    return [x_train, y_train, x_test, y_test]


'''
    处理验证数据
'''


def load_test_data(path):
    result = []
    files = os.listdir(path)
    for f in files:
        data = pd.read_csv(path + '/' + f)
        if 'TimeS' in data:
            data = data.drop(['TimeS'], axis=1)
        result.append(data)
    result = np.array(result)
    return result


'''
    模型创建函数
    使用one-hot编码后这里不能在使用sigmoid，sigmoid输出的是单概率
    损失函数在使用one-hot编码后也需要更改成categorical_crossentropy
'''


def create_model(input_shape1, input_shape2):
    model = Sequential()
    model.add(LSTM(input_shape=(input_shape1, input_shape2), units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_shape=(None, 50), units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_shape=(None, 50), units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    # 使用one-hot编码后这里不能在使用sigmoid，sigmoid输出的是单概率
    model.add(Dense(2, activation='softmax'))
    start = time.time()
    # 损失函数在使用one-hot编码后也需要更改成categorical_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    # 查看模型的结构
    model.summary()
    return model


'''
    绘图函数
'''


def draw(history):
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.xlabel('Epochs', fontsize=12)
    pyplot.ylabel('Loss', fontsize=12)
    pyplot.savefig("./images1")
    pyplot.show()
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.xlabel('Epochs', fontsize=12)
    pyplot.ylabel('Accuracy', fontsize=12)
    pyplot.savefig("./images2")
    pyplot.show()


if __name__ == '__main__':
    path1 = './故障前1500'
    path2 = './正常1500'
    path3 = './验证'
    model_path = './models/lstm_model_label.h5'

    '''
    训练时时以下解除注释
    '''
    # # 数据处理
    # X_train, y_train, X_test, y_test = load_data(path1, path2, 1, 0)
    # print('X_train shape:', X_train.shape)
    # print('y_train shape:', y_train.shape)
    # print('X_test shape:', X_test.shape)
    # print('y_test shape:', y_test.shape)
    #
    # # 模型训练
    # model = create_model(X_train.shape[1], X_train.shape[2])
    # history = model.fit(X_train, y_train, epochs=15, batch_size=4, validation_data=(X_test, y_test), verbose=2,
    #                     shuffle=False)
    # model.save(model_path)
    # draw(history)
    #
    # # 模型评估
    # score = model.evaluate(X_test, y_test, verbose=2)  # evaluate函数按batch计算在某些输入数据上模型的误差
    # print('Test accuracy:', score[1])
    # score = model.evaluate(X_train, y_train, verbose=2)  # evaluate函数按batch计算在某些输入数据上模型的误差
    # print('Train accuracy:', score[1])

    # 预测验证
    model = keras.models.load_model(model_path)
    data = load_test_data(path3)
    # print(data.shape)
    predict_x = model.predict(data)
    classes_x = np.argmax(predict_x, axis=1)
    print(classes_x)
