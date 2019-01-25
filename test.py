import os
import sys
import tkinter, tkinter.filedialog, tkinter.messagebox
from pprint import pprint
import numpy as np

import pandas as pd

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


def rot_mod(rot):
    return rot


frame = 2
times = time_data
poses = pos_data
accels = accel_data
data, target = [], []
first_poses = np.vstack((np.zeros((1, 3)), poses[0:frame - 1]))

prev_time = np.zeros((frame, 1))
prev_gyro = np.vstack((np.array([0.0, 0.0, 0.0]), gyro_data[:frame - 1, :]))
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
test_pos_input, test_pos_target = np.array(data), np.array(target)

print(test_pos_input[:, 0, :])
v = np.vstack((test_pos_input[:, 0, :], np.zeros((frame, 3))))
df['test_in_x'] = v[:, 0]
df['test_in_y'] = v[:, 1]
df['test_in_z'] = v[:, 2]
df['test_out_r'] = np.vstack((np.zeros((frame, 1)), test_pos_target.reshape((test_pos_target.shape[0], 1))))
df.to_csv('test.csv')

print('end')