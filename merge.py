import glob
import math
import os
import tkinter, tkinter.filedialog, tkinter.messagebox
from pprint import pprint

import pandas as pd
import sys
import datetime

OUTPUT = 'merged'


def get_dt(column):
    return int(column['date']) * 24 * 60 * 60 + \
           int(column['hour']) * 60 * 60 + \
           int(column['min']) * 60 + \
           float(column['sec'])

'''
# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()
fTyp = [("", "sensor*.csv")]
iDir = os.path.abspath(os.path.dirname(__file__))
sensor_file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir='H:\Research_Resource/2')

# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()
fTyp = [("", "position*.csv")]
iDir = os.path.abspath(os.path.dirname(__file__))
position_file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir='H:\Research_Resource/2')
'''

if __name__ == '__main__':

    sensor_files = glob.glob("H:\Research_Resource/4/sen*.csv")
    position_files = glob.glob("H:\Research_Resource/4/pos*.csv")
    sensor_files.sort()
    position_files.sort()

    pprint(sensor_files)
    pprint(position_files)

    for sensor_file, position_file in zip(sensor_files, position_files):
        # センサデータ読み込み & 各センサごとに分割
        sensor_input = pd.read_csv(sensor_file, encoding="utf-8")

        # 位置データ読み込み
        position_data = pd.read_csv(position_file, encoding="utf-8")

        pos_last_idx = position_data.last_valid_index()
        # 各センサの記録開始時間
        sensor_start = get_dt(sensor_input.iloc[0])

        out_data = []
        sensor_idx = 0
        finish = False
        for pos_idx in range(1, pos_last_idx):

            pos_col = position_data.iloc[pos_idx]
            pos_time = get_dt(pos_col)
            max_idx = sensor_input.size
            while get_dt(sensor_input.iloc[sensor_idx + 1]) < pos_time:
                if sensor_idx + 1 >= max_idx:
                    finish = True
                    break
                sensor_idx = sensor_idx + 1

            if finish or abs(get_dt(sensor_input.iloc[sensor_idx]) - pos_time) < abs(get_dt(sensor_input.iloc[sensor_idx + 1]) - pos_time):
                sensor_col = sensor_input.iloc[sensor_idx]
            else:
                sensor_col = sensor_input.iloc[sensor_idx + 1]

            sensor_dt = get_dt(sensor_col)
            # csv用の一行分のデータを作成
            data = list()
            data.append(pos_time)
            data.append(sensor_dt)
            data.append(round(sensor_dt - pos_time, 3))
            data.append(-sensor_col['accel_y'])  # pos座標系_x = -accel座標系_y
            data.append(-sensor_col['accel_z'])  # pos座標系_y = -accel座標系_z
            data.append(-sensor_col['accel_x'])  # pos座標系_z = -accel座標系_x
            data.append(sensor_col['gyro_x'])
            data.append(sensor_col['gyro_y'])
            data.append(sensor_col['gyro_z'])
            data.append(pos_col['pos_x'])
            data.append(pos_col['pos_y'])
            data.append(pos_col['pos_z'])
            data.append(pos_col['rot_x'])
            data.append(pos_col['rot_y'])
            out_data.append(data)
            sys.stdout.write('\r' + '結合中: {}% ({}/{})'.format(int(pos_idx / pos_last_idx * 100),
                                                              pos_idx,
                                                              pos_last_idx))
            sys.stdout.flush()

        # 出力するCSVデータの作成
        out_frame = pd.DataFrame(out_data)
        # ヘッダー(各数値の意味とか名前とかを表の一番上に入れとくやつ)設定
        out_frame.columns = ['dt[s]', 'sensor_dt[s]', 'pos-sens_delta[ms]',
                             'accel_x', 'accel_y', 'accel_z',
                             'gyro_x', 'gyro_y', 'gyro_z',
                             'pos_x', 'pos_y', 'pos_z',
                             'rot_x', 'rot_y']
        out_frame.to_csv('merged2/'+os.path.basename(position_file) + OUTPUT + ".csv", index=False)
        # out_frame.loc[:, 'pos_x':'rot_y'].to_csv(Y_OUT, index=False)
