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
from keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from keras.optimizers import RMSprop, Adam

# ファイル選択ダイアログの表示
from keras.utils import plot_model
from numpy.core.multiarray import ndarray


def make_dataset(times: np.ndarray, accels: np.ndarray, poses: np.ndarray, width: int):
    data, target = [], []
    prev_time = np.zeros((width, 1))
    first_poses = np.vstack((np.zeros((1, 3)), poses[0:width-1]))
    is_first = True
    for i in range(len(times) - width):
        time = times[i:i+width].reshape(width, 1)
        if is_first:
            data.append(np.hstack((time-prev_time, accels[i:i+width, :], (poses[i:i+width, :]-first_poses)*100)))
        else:
            data.append(np.hstack((time-prev_time, accels[i:i+width, :], (poses[i:i+width, :]-poses[i-1: i+width-1, :])*100)))
        target.append((poses[i+width] - poses[i+width-1]) * 100)
        prev_time = time
    return np.array(data), np.array(target)


def param_model():
    '''
    frame = {{choice([4, 6, 8, 10, 12, 14, 16])}}
    hide_num = {{choice([1, 2, 3])}}
    hide_unit = {{choice([4, 8, 16, 32, 64, 128])}}
    lstm_unit = {{choice([4, 8, 16, 32, 64, 128])}}
    lr = {{uniform(0, 0.00001)}}
    '''
    frame = 10
    hide_num = 1
    hide_unit = 32
    lstm_unit = 64
    lr = 0.0001

    model = Sequential()
    model.add(LSTM(lstm_unit, batch_input_shape=(None, frame, 7), return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    for _ in range(hide_num):
        model.add(Dense(hide_unit))
        model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('linear'))

    plot_model(model, to_file='model.png', show_shapes=True)
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    # es = EarlyStopping(patience=30, monitor='loss', verbose=1, mode='auto')
    es = EarlyStopping(patience=20)
    rlr = ReduceLROnPlateau()

    file_list = glob.glob("position_*.csv")

    file_num = 0
    for train_file in file_list:
        log_path = './logs/log_{}_{}_{}_{}_{}-{}/'.format(frame, hide_num, hide_unit, lstm_unit, lr, file_num)
        filepath = './saves/models_{}_{}_{}_{}_{:.0f}-{}/'.format(frame, hide_num, hide_unit, lstm_unit, lr*1000000, file_num)
        os.makedirs(filepath, exist_ok=True)
        tb = TensorBoard(log_dir=log_path)
        cp = ModelCheckpoint(filepath=filepath+'model_{epoch:02d}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        df = pd.read_csv(train_file)
        time_data = df.loc[:, 'dt[s]'].values
        pos_data = df.loc[:, 'pos_x': 'pos_z'].values
        accel_data = df.loc[:, 'accel_x': 'accel_z'].values
        gyro_data = df.loc[:, 'gyro_x': 'gyro_z'].values
        rot_data = df.loc[:, 'rot_x': 'rot_y'].values

        width = frame
        times = time_data
        poses = pos_data
        accels = accel_data
        data, target = [], []
        prev_time = np.zeros((width, 1))
        first_poses = np.vstack((np.zeros((1, 3)), poses[0:width - 1]))
        is_first = True
        for i in range(len(times) - width):
            time = times[i:i + width].reshape(width, 1)
            if is_first:
                data.append(
                    np.hstack((time - prev_time, accels[i:i + width, :], (poses[i:i + width, :] - first_poses) * 100)))
            else:
                data.append(np.hstack((time - prev_time, accels[i:i + width, :],
                                       (poses[i:i + width, :] - poses[i - 1: i + width - 1, :]) * 100)))
            target.append((poses[i + width] - poses[i + width - 1]) * 100)
            prev_time = time
        pos_input, pos_target = np.array(data), np.array(target)

        print(pos_input[:, 0, 3:7])
        model.fit(pos_input, pos_target, epochs=1000, verbose=1, batch_size=10,
                  callbacks=[tb, es, cp, rlr], validation_split=0.1,
                  shuffle=True)

    test_file = file_list[4]

    df = pd.read_csv(test_file)
    time_data = df.loc[:, 'dt[s]'].values
    pos_data = df.loc[:, 'pos_x': 'pos_z'].values
    accel_data = df.loc[:, 'accel_x': 'accel_z'].values
    gyro_data = df.loc[:, 'gyro_x': 'gyro_z'].values
    rot_data = df.loc[:, 'rot_x': 'rot_y'].values

    width = frame
    times = time_data
    poses = pos_data
    accels = accel_data
    data, target = [], []
    prev_time = np.zeros((width, 1))
    first_poses = np.vstack((np.zeros((1, 3)), poses[0:width - 1]))
    is_first = True
    for i in range(len(times) - width):
        time = times[i:i + width].reshape(width, 1)
        if is_first:
            data.append(
                np.hstack((time - prev_time, accels[i:i + width, :], (poses[i:i + width, :] - first_poses) * 100)))
        else:
            data.append(np.hstack((time - prev_time, accels[i:i + width, :],
                                   (poses[i:i + width, :] - poses[i - 1: i + width - 1, :]) * 100)))
        target.append((poses[i + width] - poses[i + width - 1]) * 100)
        prev_time = time
    # 埋め込み
    test_pos_input, test_pos_target = np.array(data), np.array(target)

    pred = np.array([[0.0, 0.0, 0.0]]*df.shape[0])
    pred[:frame, :] = test_pos_input[0, :, 4:]
    # evaluation
    total_loss = 0

    i = frame
    for pos_input in test_pos_input:
        feed_input = np.hstack((pos_input[:, 0:4], pred[i-10:i, :])).reshape((1, 10, 7))
        pred[i] = model.predict(feed_input)
        total_loss = total_loss + test_pos_target[i-frame]-pred[i]
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
                                          max_evals=100,
                                          trials=Trials())
    print(best_model.summary())
    print(best_run)
    best_model.save(filepath='best_model.hdf5')
    '''
    param_model()
    '''
