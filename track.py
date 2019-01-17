import sys

import pandas as pd
from keras import Model
from keras.engine.saving import load_model
import numpy as np

import learn

if __name__ == '__main__':
    model: Model = load_model(sys.argv[1])
    frame = 6
    (train_X, train_Y), (test_X, test_Y) = learn.create_data(sys.argv[2], frame)
    pred = model.predict(train_X)
    pred = np.array(pred)

    print(pred.shape)
    pred = pred.transpose()
    print(pred.shape)
    pred = pred.reshape((pred.shape[1], pred.shape[2]))
    print(pred.shape)

    df = pd.read_csv(sys.argv[2])
    time_pos = [np.array(df.iloc[i, 7:12]) for i in range(frame)]
    output = time_pos[frame-1]
    for row in pred:
        output = output + np.array(row)
        time_pos.append(output)
    out_df = pd.DataFrame(np.array(time_pos))
    out_df.to_csv('pos_pred.csv', index=False)


