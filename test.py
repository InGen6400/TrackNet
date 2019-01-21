import os
import sys
import tkinter, tkinter.filedialog, tkinter.messagebox
from pprint import pprint
import numpy as np

import pandas as pd

filter_alpha = 0.05


def high_pass_filter(accels: np.ndarray):
    g = [np.zeros(3)]*accels.shape[0]
    for i in range(accels.shape[0]):
        g[i][0] = (1 - filter_alpha) * g[i][0] + filter_alpha * accels[i, 0]
        g[i][1] = (1 - filter_alpha) * g[i][1] + filter_alpha * accels[i, 1]
        g[i][2] = (1 - filter_alpha) * g[i][2] + filter_alpha * accels[i, 2]

        accels[i, 0] = accels[i, 0] - g[i][0]
        accels[i, 1] = accels[i, 1] - g[i][1]
        accels[i, 2] = accels[i, 2] - g[i][2]

        g[i] = np.array(g[i])
        print(g[i])

    print(accels[:, :])
    return accels, np.array(g)


def make_dataset(times: np.ndarray, accels: np.ndarray, poses: np.ndarray, width: int):
    data, target = [], []
    prev_time = np.zeros((width, 1))
    is_first = True
    first_delta = np.vstack((np.zeros(3), poses[0:width-1, :]))
    for i in range(len(times) - width):
        time = times[i:i+width].reshape(width, 1)
        if is_first:
            data.append(np.hstack((time - prev_time, accels[i:i + width, :], (poses[i:i+width, :] - first_delta) * 100)))
            is_first = False
        else:
            data.append(np.hstack((time-prev_time, accels[i:i+width, :], (poses[i:i+width, :]-poses[i-1: i+width-1])*100)))
        target.append((poses[i+width] - poses[i+width-1]) * 100)
        prev_time = time
    return np.array(data), np.array(target)


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
accel_data, g = high_pass_filter(accel_data)
gyro_data = df.loc[:, 'gyro_x': 'gyro_z'].values
rot_data = df.loc[:, 'rot_x': 'rot_y'].values

data, target = make_dataset(time_data, accel_data, pos_data, 3)
data = np.array(data)

df['accel_high_x'] = accel_data[:, 0]
df['accel_high_y'] = accel_data[:, 1]
df['accel_high_z'] = accel_data[:, 2]
df['gx'] = g[:, 0]
df['gy'] = g[:, 1]
df['gz'] = g[:, 2]
df.to_csv('test.csv')
