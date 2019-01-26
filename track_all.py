import math
import os
import sys

import pandas as pd
from keras import Model
from keras.engine.saving import load_model
import numpy as np

import tkinter, tkinter.filedialog, tkinter.messagebox

from old_accel import get_old_method_pos

frame = 8


# ニューラルネットへ投げるためのデータセット作成
def make_dataset(times: np.ndarray, accels: np.ndarray, poses: np.ndarray, width: int):
    data, target = [], []
    first_poses = np.vstack((np.zeros((1, 3)), poses[0:width-1]))
    is_first = True

    v = np.zeros_like(accels)
    i = 0
    for a in accels:
        if i>0:
            dt = times[i] - times[i - 1]
            v[i] = v[i-1] + a * 9.8 * dt
        else:
            v[i] = a * 9.8 * times[0]
        i = i + 1

    for i in range(len(times) - width):
        time = times[i:i + width].reshape(width, 1)
        temp = v[i:i + width, :]
        data.append(temp)
        dt = times[i + width] - times[i + width - 1]
        delta = poses[i + width] - poses[i + width - 1]
        dist = math.sqrt(delta[0] * delta[0] + delta[2] * delta[2])
        temp = dist / dt
        target.append(temp)

    return np.array(data), np.array(target)


def pred2pos(dv: np.ndarray, rotation: np.ndarray, times: np.ndarray):
    # 返り値となる座標データ
    ret = np.zeros_like(dv)
    is_first = True
    for i in range(dv.shape[0]):
        # ラジアンへ変換
        rot = -rotation[i] * math.pi / 180
        # 回転行列
        R = np.array([
            [math.cos(rot), 0, math.sin(rot)],
            [0, 1, 0],
            [-math.sin(rot), 0, math.cos(rot)]
        ])
        if is_first:
            ret[0] = np.dot(dv[i], R) * times[0]
            is_first = False
        else:
            # 速度を積分で算出
            ret[i] = ret[i-1] + np.dot(dv[i], R) * (times[i] - times[i - 1])
    return ret


if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("", "merge*.csv")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    test_file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

    # csvからデータ読み込み
    df = pd.read_csv(test_file)
    time_data = df.loc[:, 'dt[s]'].values
    pos_data = df.loc[:, 'pos_x': 'pos_z'].values
    accel_data = df.loc[:, 'accel_x': 'accel_z'].values
    gyro_data = df.loc[:, 'gyro_x': 'gyro_z'].values
    rot_data = df.loc[:, 'rot_x': 'rot_y'].values

    # ニューラルモデルファイルの選択画面表示
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
        pred[i, 2] = model.predict(np.array([pos_input]))
        total_loss = total_loss + test_pos_target[i-frame]-pred[i]
        i = i + 1

    pos = pred2pos(pred, rot_data[:, 1], time_data)
    print(pos)
    #print(np.array(pos_data))

    old_method_predict = get_old_method_pos(time_data, accel_data, rot_data[:, 1])

    # csvへの書き込みと保存
    df['old_x'] = -old_method_predict[:, 0]
    df['old_y'] = old_method_predict[:, 1]
    df['old_z'] = old_method_predict[:, 2]
    df['raw_pred_x'] = pred[:, 0]
    df['raw_pred_y'] = pred[:, 1]
    df['raw_pred_z'] = pred[:, 2]
    df['pred_x'] = pos[:, 0]
    df['pred_y'] = pos[:, 1]
    df['pred_z'] = pos[:, 2]
    df.to_csv('./result/track_' + os.path.basename(test_file)[9:28] + '.csv')
