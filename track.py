import sys

import pandas as pd
from keras import Model
from keras.engine.saving import load_model
import numpy as np

import learn

if __name__ == '__main__':
    model: Model = load_model(sys.argv[1])
    frame = 4
    (train_X, train_Y), (test_X, test_Y) = learn.create_data(sys.argv[2], frame)
    print(train_X.shape)
    print(train_X[0].shape)
    output = model.predict(train_X)

    df = pd.read_csv(sys.argv[2])
