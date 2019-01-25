import glob
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
from keras.optimizers import RMSprop, Adam

import tkinter, tkinter.filedialog, tkinter.messagebox

from keras.utils import plot_model
from numpy.core.multiarray import ndarray


def param_model():
    frame = {{choice([1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24])}}
    hide_num = {{choice([1, 2, 3])}}
    hide_unit = {{choice([2, 4, 8, 16])}}
    lstm_unit = {{choice([2, 4, 8, 16])}}
    lr = {{choice([-2, -3, -4, -5, -6])}}
    lr = 10**lr
    '''
    frame = 2
    hide_num = 1
    hide_unit = 16
    lstm_unit = 16
    lr = 0.001
    '''
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
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    # es = EarlyStopping(patience=30, monitor='loss', verbose=1, mode='auto')
    es = EarlyStopping(patience=20)
    rlr = ReduceLROnPlateau(patience=10)

    file_list = glob.glob("./merged/position_*.csv")

    file_num = 0
    for train_file in file_list:
        log_path = './logs2/log_{}_{}_{}_{}_{}-{}/'.format(frame, hide_num, hide_unit, lstm_unit, lr, file_num)
        filepath = './saves2/models_{}_{}_{}_{}_{:.0f}-{}/'.format(frame, hide_num, hide_unit, lstm_unit, lr*1000000, file_num)
        os.makedirs(filepath, exist_ok=True)
        tb = TensorBoard(log_dir=log_path)
        cp = ModelCheckpoint(filepath=filepath+'model_{epoch:02d}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        df = pd.read_csv(train_file)
        time_data = df.loc[:, 'dt[s]'].values
        gyro_data = df.loc[:, 'gyro_x': 'gyro_z'].values
        rot_data = df.loc[:, 'rot_x': 'rot_y'].values

        width = frame
        times = time_data
        data, target = [], []
        prev_time = np.zeros((width, 1))
        prev_gyro = np.vstack((np.array([0.0, 0.0, 0.0]), gyro_data[:frame-1, :]))
        is_first = True
        for i in range(len(times) - frame):
            time = times[i:i + frame].reshape(frame, 1)
            DT = (time - prev_time)
            if is_first:
                temp = np.abs((prev_gyro - gyro_data[i:i + frame, :] + 180) % 360)
                is_first = False
            else:
                temp = np.abs((gyro_data[i - 1:i + frame - 1, :] - gyro_data[i:i + frame, :] + 180) % 360)
            data.append((temp - 180) / DT)
            dt = times[i + frame] - times[i + frame - 1]
            target.append((np.abs((rot_data[i + frame, 1] - rot_data[i + frame - 1, 1] + 180) % 360) - 180) / dt)
            prev_time = time
        dr_input, w_target = np.array(data), np.array(target)

        print(dr_input[:, 0, 3:7])
        model.fit(dr_input, w_target, epochs=1000, verbose=1, batch_size=10,
                  callbacks=[tb, es, cp, rlr], validation_split=0.1,
                  shuffle=True)
        file_num = file_num + 1

    test_file = file_list[4]

    df = pd.read_csv(test_file)
    time_data = df.loc[:, 'dt[s]'].values
    gyro_data = df.loc[:, 'gyro_x': 'gyro_z'].values
    rot_data = df.loc[:, 'rot_x': 'rot_y'].values

    width = frame
    times = time_data
    data, target = [], []
    prev_time = np.zeros((width, 1))
    prev_gyro = np.vstack((np.array([0.0, 0.0, 0.0]), gyro_data[:frame-1, :]))
    is_first = True
    for i in range(len(times) - frame):
        time = times[i:i + frame].reshape(frame, 1)
        DT = (time - prev_time)
        if is_first:
            temp = np.abs((prev_gyro - gyro_data[i:i + frame, :] + 180) % 360)
            is_first = False
        else:
            temp = np.abs((gyro_data[i - 1:i + frame - 1, :] - gyro_data[i:i + frame, :] + 180) % 360)
        data.append((temp - 180) / DT)
        dt = times[i + frame] - times[i + frame - 1]
        target.append((np.abs((rot_data[i + frame, 1] - rot_data[i + frame - 1, 1] + 180) % 360) - 180) / dt)
        prev_time = time

    test_pos_input, test_pos_target = np.array(data), np.array(target)

    pred = np.array([[0.0, 0.0, 0.0]]*df.shape[0])
    pred[:frame, :] = test_pos_input[0, :, :]
    # evaluation
    total_loss = 0

    i = frame
    for dr_input in test_pos_input:
        pred[i] = model.predict(np.array([dr_input]))
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
    best_run, best_model = optim.minimize(model=param_model,
                                          data=dummy,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials(),
                                          eval_space=True)
    print(best_model.summary())
    print(best_run)
    best_model.save(filepath='best_gyro_model.hdf5')
    '''
    param_model()

    '''
