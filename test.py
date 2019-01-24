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

width = 2
times = time_data
poses = pos_data
accels = accel_data
data, target = [], []
prev_time = np.array([[0], [times[0]]])
first_poses = np.vstack((np.zeros((1, 3)), poses[0:width - 1]))
is_first = True

for i in range(len(times) - width):
    time = times[i:i + width].reshape(width, 1)
    DT = (time - prev_time)
    if is_first:
        data.append(
            np.hstack((accels[0:width, :] * 9.8 * DT, (poses[0:width, :] - first_poses) / DT)))
        is_first = False
    else:
        data.append(np.hstack((accels[i:i + width, :] * 9.8 * DT,
                               (poses[i:i + width, :] - poses[i - 1: i + width - 1, :]) / DT)))
    dt = times[i + width] - times[i + width - 1]
    target.append((poses[i + width] - poses[i + width - 1]) / dt)
    prev_time = time

test_pos_input, test_pos_target = np.array(data), np.array(target)

print(test_pos_input[:, 0, :])
v = np.vstack((test_pos_input[:, 0, :], np.zeros((width, 6))))
df['test_at_x'] = v[:, 0]
df['test_at_y'] = v[:, 1]
df['test_at_z'] = v[:, 2]
df['test_v_x'] = v[:, 3]
df['test_v_y'] = v[:, 4]
df['test_v_z'] = v[:, 5]
df.to_csv('test.csv')

print('end')