import math
import os
import pandas as pd
import numpy as np
import tkinter, tkinter.filedialog, tkinter.messagebox


def get_old_method_pos(times, accels, rots):
    pred = np.zeros((times.shape[0], 3))
    v = np.array([0, 0, 0])
    for i in range(len(times)):
        # ラジアンへ変換
        rot = rots[i] * math.pi / 180
        # 回転行列
        R = np.array([
            [math.cos(rot), 0, math.sin(rot)],
            [0, 1, 0],
            [-math.sin(rot), 0, math.cos(rot)]
        ])
        if i == 0:
            dt = times[0]
        else:
            dt = times[i] - times[i-1]
        v = v + accels[i] * 9.8 * dt
        if i == 0:
            pred[i] = np.dot(v, R) * dt
        else:
            pred[i] = pred[i-1] + np.dot(v, R) * dt
        i = i + 1
        if i+1 >= times.shape[0]:
            break
    return pred


if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("", "merge*.csv")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

    df = pd.read_csv(file)
    time_data = df.loc[:, 'dt[s]'].values
    accel_data = df.loc[:, 'accel_x': 'accel_z'].values

    accel_pred = get_old_method_pos(time_data, accel_data)

    df['accel_pred_x'] = accel_pred[:, 0]
    df['accel_pred_y'] = accel_pred[:, 1]
    df['accel_pred_z'] = accel_pred[:, 2]

    df.to_csv('old_accel2.csv')
