import os
import sys

import pandas as pd
from keras import Model
from keras.engine.saving import load_model
import numpy as np

import tkinter, tkinter.filedialog, tkinter.messagebox

from old_accel import get_old_method_pos

frame = 12


def make_dataset(times: np.ndarray, gyro_data: np.ndarray, width: int):
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

    return np.array(data), np.array(target)


def pred2rot(pred: np.ndarray, times: np.ndarray):
    ret = np.zeros_like(pred)
    is_first = True
    for i in range(pred.shape[0]):
        if is_first:
            ret[0] = pred[0] * times[0]
            is_first = False
        else:
            ret[i] = ret[i-1] + pred[i] * (times[i]-times[i-1])
    return ret


if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("", "merge*.csv")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    test_file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

    df = pd.read_csv(test_file)
    time_data = df.loc[:, 'dt[s]'].values
    gyro_data = df.loc[:, 'gyro_x': 'gyro_z'].values
    rot_data = df.loc[:, 'rot_x': 'rot_y'].values

    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("", "model*.hdf5")]
    iDir = os.path.abspath(os.path.dirname(__file__+'saves'))
    model_file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

    test_gyro_input, test_pos_target = make_dataset(time_data, gyro_data, frame)

    model = load_model(model_file)

    pred = np.array([0.0]*df.shape[0])
    # evaluation
    total_loss = 0
    i = frame
    for pos_input in test_gyro_input:
        pred[i] = model.predict(np.array([pos_input]))
        total_loss = total_loss + test_pos_target[i-frame]-pred[i]
        i = i + 1

    print(pred)

    rot = pred2rot(pred, time_data)
    print(rot)
    #print(np.array(pos_data))

    df['pred_rot'] = rot
    df['pred_w'] = pred
    df.to_csv('track_gyro_out.csv')
