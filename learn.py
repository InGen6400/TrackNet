import os
import sys

import pandas as pd
import numpy as np
from hyperas import optim
from hyperopt import STATUS_OK, tpe, Trials
from keras import Sequential, Input, Model
from hyperas.distributions import choice, uniform
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import RMSprop, Adam


def dummy():
    return


def create_data(file, frame: int):
    df = pd.read_csv(file)

    train_X = []
    test_X = []
    train_pos_x = []
    train_pos_y = []
    train_pos_z = []
    train_rot_x = []
    train_rot_y = []
    test_Y = []
    test_pos_x = []
    test_pos_y = []
    test_pos_z = []
    test_rot_x = []
    test_rot_y = []
    train = df.copy()
    test = df.iloc[10:500, :].copy()

    for i in range(len(train) - frame):
        temp = train[i:(i+frame)].copy()
        train_X.append(temp.loc[:, 'accel_x':'gyro_z'].values)
        y_tmp = temp.iloc[2, 7:12] - temp.iloc[1, 7:12]
        train_pos_x.append(y_tmp['pos_x'])
        train_pos_y.append(y_tmp['pos_y'])
        train_pos_z.append(y_tmp['pos_z'])
        train_rot_x.append(y_tmp['rot_x'])
        train_rot_y.append(y_tmp['rot_y'])

    for i in range(len(test) - frame):
        temp = test[i:(i + frame)].copy()
        test_X.append(temp.loc[:, 'accel_x':'gyro_z'])
        y_tmp = temp.iloc[2, 7:12] - temp.iloc[1, 7:12]
        test_pos_x.append(y_tmp['pos_x'])
        test_pos_y.append(y_tmp['pos_y'])
        test_pos_z.append(y_tmp['pos_z'])
        test_rot_x.append(y_tmp['rot_x'])
        test_rot_y.append(y_tmp['rot_y'])

    train_X = [np.array(train_x_input) for train_x_input in train_X]
    train_X = np.array(train_X)
    train_Y = {
        'pos_x': np.array(train_pos_x),
        'pos_y': np.array(train_pos_y),
        'pos_z': np.array(train_pos_z),
        'rot_x': np.array(train_pos_x),
        'rot_y': np.array(train_pos_y),
    }

    test_X = [np.array(test_x_input) for test_x_input in test_X]
    test_X = np.array(test_X)
    test_Y = {
        'pos_x': np.array(test_pos_x),
        'pos_y': np.array(test_pos_y),
        'pos_z': np.array(test_pos_z),
        'rot_x': np.array(test_pos_x),
        'rot_y': np.array(test_pos_y),
    }

    return (train_X, train_Y), (test_X, test_Y)


def param_model():
    frame = {{choice([2, 4, 6, 8, 10, 12, 14])}}
    hide_num = {{choice([1, 2, 3, 4])}}
    hide_unit = {{choice([32, 64, 128, 256])}}
    dense_unit = {{choice([32, 64, 128, 256])}}
    lr = {{uniform(0, 0.0001)}}
    '''
    frame = 6
    hide_num = 2
    hide_unit = 128
    dense_unit = 128
    lr = 0.0001
    '''

    df = pd.read_csv(sys.argv[1])

    train_X = []
    test_X = []
    train_pos_x = []
    train_pos_y = []
    train_pos_z = []
    train_rot_x = []
    train_rot_y = []
    test_Y = []
    test_pos_x = []
    test_pos_y = []
    test_pos_z = []
    test_rot_x = []
    test_rot_y = []
    train = df.copy()
    test = df.iloc[10:500, :].copy()

    for i in range(len(train) - frame):
        temp = train[i:(i+frame)].copy()
        train_X.append(temp.loc[:, 'accel_x':'gyro_z'].values)
        y_tmp = temp.iloc[2, 7:12] - temp.iloc[1, 7:12]
        train_pos_x.append(y_tmp['pos_x'])
        train_pos_y.append(y_tmp['pos_y'])
        train_pos_z.append(y_tmp['pos_z'])
        train_rot_x.append(y_tmp['rot_x'])
        train_rot_y.append(y_tmp['rot_y'])

    for i in range(len(test) - frame):
        temp = test[i:(i + frame)].copy()
        test_X.append(temp.loc[:, 'accel_x':'gyro_z'])
        y_tmp = temp.iloc[2, 7:12] - temp.iloc[1, 7:12]
        test_pos_x.append(y_tmp['pos_x'])
        test_pos_y.append(y_tmp['pos_y'])
        test_pos_z.append(y_tmp['pos_z'])
        test_rot_x.append(y_tmp['rot_x'])
        test_rot_y.append(y_tmp['rot_y'])

    train_X = [np.array(train_x_input) for train_x_input in train_X]
    train_X = np.array(train_X)
    train_Y = {
        'pos_x': np.array(train_pos_x),
        'pos_y': np.array(train_pos_y),
        'pos_z': np.array(train_pos_z),
        'rot_x': np.array(train_pos_x),
        'rot_y': np.array(train_pos_y),
    }

    test_X = [np.array(test_x_input) for test_x_input in test_X]
    test_X = np.array(test_X)
    test_Y = {
        'pos_x': np.array(test_pos_x),
        'pos_y': np.array(test_pos_y),
        'pos_z': np.array(test_pos_z),
        'rot_x': np.array(test_pos_x),
        'rot_y': np.array(test_pos_y),
    }

    print(train_X.shape)
    '''
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
    '''
    input_layer = Input(shape=(train_X.shape[1], train_X.shape[2]))
    x = Flatten()(input_layer)
    x = Dense(dense_unit, activation='relu')(x)
    for _ in range(hide_num):
        x = Dense(hide_unit, activation='relu')(x)
    output_pos_x = Dense(1, activation='linear', name='pos_x')(x)
    output_pos_y = Dense(1, activation='linear', name='pos_y')(x)
    output_pos_z = Dense(1, activation='linear', name='pos_z')(x)
    output_rot_x = Dense(1, activation='linear', name='rot_x')(x)
    output_rot_y = Dense(1, activation='linear', name='rot_y')(x)
    model = Model(input_layer, [output_pos_x, output_pos_y, output_pos_z, output_rot_x, output_rot_y])
    model.summary()
    model.compile(loss={'pos_x': 'mean_squared_error',
                        'pos_y': 'mean_squared_error',
                        'pos_z': 'mean_squared_error',
                        'rot_x': 'mean_squared_error',
                        'rot_y': 'mean_squared_error'},
                  optimizer=Adam(lr=lr),
                  metrics=['mae'])

    filepath = './saves/models_{}_{}_{}_{}_{:.0f}/'.format(frame, hide_num, hide_unit, dense_unit, lr*1000000)
    os.makedirs(filepath, exist_ok=True)
    log_path = './logs/log_{}_{}_{}_{}_{}/'.format(frame, hide_num, hide_unit, dense_unit, lr)
    es = EarlyStopping(patience=30, monitor='loss', verbose=1, mode='auto')
    cp = ModelCheckpoint(filepath=filepath+'model_{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='auto')
    tb = TensorBoard(log_dir=log_path)

    model.fit(train_X, train_Y, epochs=1000, verbose=1,
              callbacks=[tb, es, cp],
              shuffle=True)

    scores = model.evaluate(test_X, test_Y, verbose=1)
    print("total loss:\t{0}".format(scores[0]))
    print("label1 loss:\t{0}\n\taccuracy:\t{1}%".format(scores[1],scores[6]))
    print("label2 loss:\t{0}\n\taccuracy:\t{1}%".format(scores[2],scores[7]))
    print("label3 loss:\t{0}\n\taccuracy:\t{1}%".format(scores[3],scores[8]))
    print("label4 loss:\t{0}\n\taccuracy:\t{1}%".format(scores[4],scores[9]))
    print("label5 loss:\t{0}\n\taccuracy:\t{1}%".format(scores[5],scores[10]))

    return {'loss': scores[0], 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=param_model,
                                          data=dummy,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
    print(best_model.summary())
    print(best_run)
    param_model()
