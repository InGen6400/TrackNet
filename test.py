import os
import sys
import tkinter, tkinter.filedialog, tkinter.messagebox
from pprint import pprint
import numpy as np

import pandas as pd


def make_dataset(times: np.ndarray, accels: np.ndarray, poses: np.ndarray, width: int):
    data, target = [], []
    prev_time = np.zeros((width, 1))
    for i in range(len(times) - width):
        time = times[i:i+width].reshape(width, 1)
        data.append(np.hstack((time-prev_time, accels[i:i+width, :], poses[i:i+width, :]*100)))
        target.append(poses[i+width]*100)
        prev_time = time
    return data, target


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

data, target = make_dataset(time_data, accel_data, pos_data, 3)
data = np.array(data)

print(data[1])
