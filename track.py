import os
import sys

import pandas as pd
from keras import Model
from keras.engine.saving import load_model
import numpy as np

import tkinter, tkinter.filedialog, tkinter.messagebox


frame = 10


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


def pred2pos(pred: np.ndarray):
    ret = np.zeros_like(pred)
    for i in range(pred.shape[0]):
        for j in range(i+1):
            ret[i, 0] = ret[i, 0] + pred[j, 0]
            ret[i, 1] = ret[i, 1] + pred[j, 1]
            ret[i, 2] = ret[i, 2] + pred[j, 2]
    return ret


if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("", "merge*.csv")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    test_file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

    df = pd.read_csv(test_file)
    time_data = df.loc[:, 'dt[s]'].values
    pos_data = df.loc[:, 'pos_x': 'pos_z'].values
    accel_data = df.loc[:, 'accel_x': 'accel_z'].values
    gyro_data = df.loc[:, 'gyro_x': 'gyro_z'].values
    rot_data = df.loc[:, 'rot_x': 'rot_y'].values

    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("", "model*.hdf5")]
    iDir = os.path.abspath(os.path.dirname(__file__+'saves'))
    model_file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

    test_pos_input, test_pos_target = make_dataset(time_data, accel_data, pos_data, frame)

    model = load_model(model_file)
    pred = np.array([[0.0, 0.0, 0.0]]*df.shape[0])
    pred[:frame, :] = test_pos_input[0, :, 4:]
    i = frame
    for pos_input in test_pos_input:
        feed_input = np.hstack((pos_input[:, 0:4], pred[i-10:i, :])).reshape((1, 10, 7))
        pred[i] = model.predict(feed_input)
        i = i + 1

    print(pred)

    first_poses = np.vstack((np.zeros((1, 3)), pos_data[0:frame - 1]))
    pred = np.vstack((pos_data[0:frame] - first_poses, pred[10:] / 100))

    pos = pred2pos(pred)
    print(pos)
    #print(np.array(pos_data))

    df['pred_x'] = pos[:, 0]
    df['pred_y'] = pos[:, 1]
    df['pred_z'] = pos[:, 2]
    df.to_csv('track_out.csv')
