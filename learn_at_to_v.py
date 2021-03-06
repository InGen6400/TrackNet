import glob
import math
import os
import random
import sys
from typing import List

import pandas as pd
import numpy as np
from hyperas import optim
from hyperopt import STATUS_OK, tpe, Trials
from keras import Sequential, Input, Model
from hyperas.distributions import choice, uniform
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Activation, Flatten, LSTM, Dropout, regularizers

import tkinter, tkinter.filedialog, tkinter.messagebox

from keras.utils import plot_model
from numpy.core.multiarray import ndarray


# TPEを用いるために外部関数を用いないようにすべてこの関数に書き込む(見づらい！！)
def param_model():
    '''
    frame = {{choice([1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24])}}
    hide_num = {{choice([1, 2, 3])}}
    hide_unit = {{choice([2, 4, 8, 16])}}
    lstm_unit = {{choice([2, 4, 8, 16])}}
    lr = {{choice([-2, -3, -4, -5, -6])}}
# {'frame': 1, 'hide_num': 1, 'hide_unit': 8, 'hide_unit_1': 8, 'lr': -4}
    lr = 10**lr
    '''
    frame = 8
    hide_num = 2
    hide_unit = 8
    lstm_unit = 8
    lr = 0.0001
    l2 = 0.0005

    model = Sequential()
    # model.add(LSTM(lstm_unit, batch_input_shape=(None, frame, 6), return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
    # model.add(Dropout(0.5, batch_input_shape=(None, frame, 6)))
    model.add(Dense(lstm_unit, batch_input_shape=(None, frame, 3)))
    model.add(Flatten())
    model.add(Activation('relu'))
    for _ in range(hide_num):
        model.add(Dense(hide_unit))
        model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))

    plot_model(model, to_file='model.png', show_shapes=True)
    model.compile(loss="mean_squared_error", optimizer="adam")

    # es = EarlyStopping(patience=30, monitor='loss', verbose=1, mode='auto')
    es = EarlyStopping(patience=5)
    rlr = ReduceLROnPlateau()

    file_list = glob.glob("./roted_data/rot*.csv")

    file_num = 0
    for train_file in file_list:
        log_path = './logs_curve/log_{}_{}_{}_{}_{}-{}/'.format(frame, hide_num, hide_unit, lstm_unit, lr, file_num)
        filepath = './saves_curve/models_{}_{}_{}_{}_{:.0f}-{}/'.format(frame, hide_num, hide_unit, lstm_unit, lr*1000000, file_num)
        os.makedirs(filepath, exist_ok=True)
        tb = TensorBoard(log_dir=log_path)
        cp = ModelCheckpoint(filepath=filepath+'model_{epoch:02d}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        df = pd.read_csv(train_file)
        time_data = df.loc[:, 'dt[s]'].values
        pos_data = df.loc[:, 'pos_x': 'pos_z'].values
        accel_data = df.loc[:, 'accel_x': 'accel_z'].values

        width = frame
        times = time_data
        poses = pos_data
        accels = accel_data
        data, target = [], []
        prev_time = np.vstack((np.zeros((1, 1)), times[:frame-1].reshape(frame-1, 1)))
        v = np.zeros_like(accels)
        i = 0
        for a in accels:
            if i>0:
                dt = times[i] - times[i - 1]
                v[i] = v[i-1] + a * 9.8 * dt
            else:
                v[i] = a * 9.8 * times[0]
            i = i + 1

        prev_time = np.vstack((np.zeros((1, 1)), times[:frame - 1].reshape(frame - 1, 1)))
        for i in range(len(times) - width):
            time = times[i:i + width].reshape(width, 1)
            DT = (time - prev_time)
            temp = v[i:i + width, :]
            data.append(temp)
            dt = times[i + width] - times[i + width - 1]
            delta = poses[i + width] - poses[i + width - 1]
            dist = math.sqrt(delta[0] * delta[0] + delta[2]*delta[2])
            temp = dist / dt
            target.append(temp)
            prev_time = time
        pos_input, pos_target = np.array(data), np.array(target)

        print(pos_input[:, 0, 3:7])
        model.fit(pos_input, pos_target, epochs=1000, verbose=1, batch_size=10,
                  callbacks=[tb, es, cp], validation_split=0.1,
                  shuffle=True)
        file_num = file_num + 1

    test_file = file_list[2]

    df = pd.read_csv(test_file)
    time_data = df.loc[:, 'dt[s]'].values
    pos_data = df.loc[:, 'pos_x': 'pos_z'].values
    accel_data = df.loc[:, 'accel_x': 'accel_z'].values

    width = frame
    times = time_data
    poses = pos_data
    accels = accel_data
    data, target = [], []
    prev_time = np.vstack((np.zeros((1, 1)), times[:frame-1].reshape(frame-1, 1)))
    v = np.zeros_like(accels)
    i = 0
    for a in accels:
        if i>0:
            dt = times[i] - times[i - 1]
            v[i] = v[i-1] + a * 9.8 * dt
        else:
            v[i] = a * 9.8 * times[0]
        i = i + 1

    prev_time = np.vstack((np.zeros((1, 1)), times[:frame - 1].reshape(frame - 1, 1)))
    for i in range(len(times) - width):
        time = times[i:i + width].reshape(width, 1)
        temp = v[i:i + width, :]
        data.append(temp)
        dt = times[i + width] - times[i + width - 1]
        delta = poses[i + width] - poses[i + width - 1]
        dist = math.sqrt(delta[0] * delta[0] + delta[2]*delta[2])
        temp = dist / dt
        target.append(temp)
        prev_time = time
    test_pos_input, test_pos_target = np.array(data), np.array(target)

    pred = np.array([[0.0]]*df.shape[0])
    pred[:frame] = test_pos_input[0, :]
    # evaluation
    total_loss = 0

    i = frame
    for pos_input in test_pos_input:
        pred[i] = model.predict(np.array([pos_input]))
        total_loss = total_loss + abs(test_pos_target[i-frame]-pred[i])
        i = i + 1

    pred = pred[frame:]
    total_loss = total_loss/test_pos_input.shape[0]
    print(test_pos_target.shape)
    print(pred.shape)
    print("total loss:\t{0}".format(total_loss))
    print("total loss sum:\t{0}".format(np.sum(total_loss)))

    return {'loss': np.sum(total_loss), 'status': STATUS_OK, 'model': model}


def dummy():
    return


if __name__ == '__main__':
    '''
    best_run, best_model = optim.minimize(model=param_model,
                                          data=dummy,
                                          algo=tpe.suggest,
                                          max_evals=25,
                                          trials=Trials(),
                                          eval_space=True)
    print(best_model.summary())
    print(best_run)
    best_model.save(filepath='best_curve_model.hdf5')
    '''
    param_model()
