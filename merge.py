import math

import pandas as pd
import sys
import datetime

OUTPUT = 'merged.csv'


def get_dt(column):
    return int(column['date']) * 24 * 60 * 60 + \
           int(column['hour']) * 60 * 60 + \
           int(column['min']) * 60 + \
           float(column['sec'])


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python merge.py sensor.csv position.csv')
        exit(-1)

    # センサデータ読み込み & 各センサごとに分割
    sensor_input = pd.read_csv(filepath_or_buffer=sys.argv[1], encoding="utf-8")

    # 位置データ読み込み
    position_data = pd.read_csv(filepath_or_buffer=sys.argv[2], encoding="utf-8")

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
        data.append(sensor_col['accel_x'])
        data.append(sensor_col['accel_y'])
        data.append(sensor_col['accel_z'])
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
    out_frame.to_csv(OUTPUT, index=False)
    # out_frame.loc[:, 'pos_x':'rot_y'].to_csv(Y_OUT, index=False)
