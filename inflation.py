import glob
import math
import os

import pandas as pd
import numpy as np


def rotate(vector: np.ndarray, theta: float):
    # 返り値となる座標データ
    ret = np.zeros_like(vector)
    # ラジアンへ変換
    rot = theta * math.pi / 180
    # 回転行列
    R = np.array([
        [math.cos(rot), 0, math.sin(rot)],
        [0, 1, 0],
        [-math.sin(rot), 0, math.cos(rot)]
    ])
    for i in range(vector.shape[0]):
        ret[i] = np.dot(vector[i], R)
    return ret


file_list = glob.glob("./merged/position_*.csv")

for file in file_list:
    # csvからデータ読み込み
    df = pd.read_csv(file)
    time_data = df.loc[:, 'dt[s]'].values
    pos_data = df.loc[:, 'pos_x': 'pos_z'].values
    accel_data = df.loc[:, 'accel_x': 'accel_z'].values
    gyro_data = df.loc[:, 'gyro_x': 'gyro_z'].values
    rot_data = df.loc[:, 'rot_x': 'rot_y'].values

    rotated_df: pd.DataFrame = df.copy()
    # 0~360 15度ごと
    for theta in range(0, 360, 15):
        rotated_accel = rotate(accel_data, theta)
        rotated_df['accel_x'] = rotated_accel[:, 0]
        rotated_df['accel_y'] = rotated_accel[:, 1]
        rotated_df['accel_z'] = rotated_accel[:, 2]

        rotated_pos = rotate(pos_data, theta)
        rotated_df['pos_x'] = rotated_pos[:, 0]
        rotated_df['pos_y'] = rotated_pos[:, 1]
        rotated_df['pos_z'] = rotated_pos[:, 2]

        rotated_df.to_csv('./roted_data/rot' + str(theta) + '_' + os.path.basename(file)[9:28] + '.csv')

