import sys

import pandas as pd
from keras import Sequential
from hyperas.distributions import choice, uniform
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

INPUT_SHAPE = 3 + 3


def param_model():
    frame = {{choice([2, 4, 6, 8, 10])}}
    hide_num = {{choice([1, 2, 3, 4])}}
    hide_unit = {{choice([32, 64, 128, 256])}}
    dense_unit = {{choice([32, 64, 128, 256])}}
    lr = {{uniform(0, 0.1)}}

    model = Sequential()
    model.add(Dense(dense_unit, input_shape=(frame, INPUT_SHAPE)))
    model.add(Activation('relu'))
    for hide in range(hide_num):
        model.add(Dense(hide_unit))
        model.add(Activation('relu'))
    model.add(Dense(3+2))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer=RMSprop())
    return model


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])

    print('Loaded data:')
    print(df)

