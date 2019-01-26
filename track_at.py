import os
import sys

import pandas as pd
from keras import Model
from keras.engine.saving import load_model
import numpy as np

import tkinter, tkinter.filedialog, tkinter.messagebox

from old_accel import get_old_method_pos

frame = 24


def make_dataset(times: np.ndarray, accels: np.ndarray, poses: np.ndarray, width: int):
    data, target = [], []
    prev_time = np.zeros((width, 1))
    first_poses = np.vstack((np.zeros((1, 3)), poses[0:width-1]))
    is_first = True

    for i in range(len(times) - width):
        time = times[i:i + width].reshape(width, 1)
        DT = (time - prev_time)
        data.append(accels[i:i + width, :] * 9.8 * DT)
        dt = times[i + width] - times[i+width-1]
        target.append((poses[i + width] - poses[i + width - 1]) / dt)
        prev_time = time

    return np.array(data), np.array(target)


def pred2pos(pred: np.ndarray, times: np.ndarray):
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
    pred[:frame, :] = test_pos_input[0, :, :]
    # evaluation
    total_loss = 0
    i = frame
    for pos_input in test_pos_input:
        pred[i] = model.predict(np.array([pos_input]))
        total_loss = total_loss + test_pos_target[i-frame]-pred[i]
        i = i + 1

    print(pred)

    first_poses = np.vstack((np.zeros((1, 3)), pos_data[0:frame - 1]))
    pred = np.vstack((pos_data[0:frame] - first_poses, pred[frame:]))

    pos = pred2pos(pred, time_data)
    print(pos)
    #print(np.array(pos_data))

    old_method_predict = get_old_method_pos(time_data, accel_data)

    df['old_x'] = old_method_predict[:, 0]
    df['old_y'] = old_method_predict[:, 1]
    df['old_z'] = old_method_predict[:, 2]
    df['pred_x'] = pos[:, 0]
    df['pred_y'] = pos[:, 1]
    df['pred_z'] = pos[:, 2]
    df['pred_vx'] = pred[:, 0]
    df['pred_vy'] = pred[:, 1]
    df['pred_vz'] = pred[:, 2]
    df.to_csv('track_out.csv')
