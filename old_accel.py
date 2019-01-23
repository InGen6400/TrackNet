
import os
import pandas as pd
import numpy as np
import tkinter, tkinter.filedialog, tkinter.messagebox


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

accel_pred = np.zeros((df.shape[0], 3))
accel_pred[0] = np.array(pos_data[0])
prev_time = 0
v = np.array([0, 0, 0])
i = 0


for time, accel in zip(time_data, accel_data):
    dt = time-prev_time
    v = v + accel * 9.8 * dt
    accel_pred[i+1] = accel_pred[i] + v * dt
    prev_time = time
    i = i + 1
    if i+1 >= df.shape[0]:
        break

df['accel_pred_x'] = accel_pred[:, 0]
df['accel_pred_y'] = accel_pred[:, 1]
df['accel_pred_z'] = accel_pred[:, 2]

df.to_csv('old_accel.csv')