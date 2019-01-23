import os
import tkinter, tkinter.filedialog, tkinter.messagebox
import pandas as pd
import numpy as np


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

noize_coeff = 1000

accel_data = accel_data + np.random.randn(accel_data.shape[0], 3) / noize_coeff
df['accel_x'] = accel_data[:, 0]
df['accel_y'] = accel_data[:, 1]
df['accel_z'] = accel_data[:, 2]
df.to_csv(test_file + '_noized' + noize_coeff)
