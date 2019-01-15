import sys

import pandas as pd
import numpy as np
from hyperas import optim
from hyperopt import STATUS_OK, tpe, Trials
from keras import Sequential
from hyperas.distributions import choice, uniform
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import RMSprop, Adam


def dummy():
    return


def create_data(train: pd.DataFrame, test: pd.DataFrame, frame_length: int):
    train_X = []
    test_X = []
    train_Y = []
    test_Y = []

    for i in range(len(train) - frame_length):
        temp = train[i:(i+frame_length)].copy()
        train_X.append(temp.loc[:, 'accel_x':'gyro_z'])
        train_Y.append(temp.iloc[1, 7:12] - temp.iloc[0, 7:12])

    for i in range(len(test) - frame_length):
        temp = test[i:(i + frame_length)].copy()
        test_X.append(temp.loc[:, 'accel_x':'gyro_z'])
        test_Y.append(temp.iloc[1, 7:12] - temp.iloc[0, 7:12])

    return (train_X, train_Y), (test_X, test_Y)


def param_model():
    frame = {{choice([2, 4, 6, 8, 10, 12, 14])}}
    hide_num = {{choice([1, 2, 3, 4])}}
    hide_unit = {{choice([32, 64, 128, 256])}}
    dense_unit = {{choice([32, 64, 128, 256])}}
    lr = {{uniform(0, 0.001)}}

    file = "./logs/log_{}_{}_{}_{}_{}".format(frame, hide_num, hide_unit, dense_unit, lr)

    df = pd.read_csv(sys.argv[1])

    train_X = []
    test_X = []
    train_Y = []
    test_Y = []
    frame_length = frame
    train = df.copy()
    test = df.iloc[10:500, :].copy()

    for i in range(len(train) - frame_length):
        temp = train[i:(i+frame_length)].copy()
        train_X.append(temp.loc[:, 'accel_x':'gyro_z'].values)
        train_Y.append((temp.iloc[2, 7:12] - temp.iloc[1, 7:12]).values)

    for i in range(len(test) - frame_length):
        temp = test[i:(i + frame_length)].copy()
        test_X.append(temp.loc[:, 'accel_x':'gyro_z'])
        test_Y.append(temp.iloc[2, 7:12] - temp.iloc[1, 7:12])

    train_X = [np.array(train_x_input) for train_x_input in train_X]
    train_X = np.array(train_X)
    train_Y = [np.array(train_y_input) for train_y_input in train_Y]
    train_Y = np.array(train_Y)

    test_X = [np.array(test_x_input) for test_x_input in test_X]
    test_X = np.array(test_X)
    test_Y = [np.array(test_y_input) for test_y_input in test_Y]
    test_Y = np.array(test_Y)

    print(train_X.shape)
    print(train_Y.shape)

    model = Sequential()
    model.add(Flatten(input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Activation('relu'))
    model.add(Dense(dense_unit))
    model.add(Activation('relu'))
    for hide in range(hide_num):
        model.add(Dense(hide_unit))
        model.add(Activation('relu'))
    model.add(Dense(train_Y.shape[1]))
    model.add(Activation('linear'))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))

    model.fit(train_X, train_Y, epochs=100000, verbose=1,
              callbacks=[TensorBoard(), EarlyStopping(patience=30, monitor='loss')],
              shuffle=True)

    loss = model.evaluate(test_X, test_Y, verbose=1)

    return {'loss': loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=param_model,
                                          data=dummy,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
    print(best_model.summary())
    print(best_run)

    param_model()
