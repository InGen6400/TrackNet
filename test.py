import os
import sys
import tkinter, tkinter.filedialog, tkinter.messagebox
from pprint import pprint
import numpy as np

import pandas as pd


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


# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()
fTyp = [("", "merge*.csv")]
iDir = os.path.abspath(os.path.dirname(__file__))
file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

df = pd.read_csv(file)
time_data = df.loc[:, 'dt[s]'].values
pos_data = df.loc[:, 'pos_x': 'pos_z'].values
accel_data = df.loc[:, 'accel_x': 'accel_z'].values
gyro_data = df.loc[:, 'gyro_x': 'gyro_z'].values
rot_data = df.loc[:, 'rot_x': 'rot_y'].values
frame = 3
data, target = make_dataset(time_data, accel_data, pos_data, frame)

first_poses = np.vstack((np.zeros((1, 3)), pos_data[0:frame - 1]))
target = np.vstack((pos_data[0:frame] - first_poses, target/100))

print(target)
pos = pred2pos(target)

df['accel_high_x'] = pos[:, 0]
df['accel_high_y'] = pos[:, 1]
df['accel_high_z'] = pos[:, 2]
df.to_csv('test.csv')
