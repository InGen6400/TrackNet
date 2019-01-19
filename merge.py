import math

import pandas as pd
import sys
import datetime

OUTPUT = 'merged.csv'
Y_OUT = 'pos.csv'


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
    prev_time = None
    for pos_idx in range(pos_last_idx):

        pos = position_data.iloc[pos_idx]
        pos_time = get_time_pos(pos)
        if prev_time is not None:
            delta_time = pos_time - prev_time
        else:
            delta_time = pos_time - pos_time

        gyro = None
        # ジャイロセンサのインデックスを位置データの時刻に合わせる
        for _, gyro in gyro_data.iterrows():
            if get_time_sensor(gyro) > pos_time:
                break
        gyro_dt = get_time_sensor(gyro) - pos_time
        accel = None
        # 加速度センサのインデックスを位置データの時刻に合わせる
        for _, accel in acceleration_data.iterrows():
            if get_time_sensor(accel) > pos_time:
                break
        accel_dt = get_time_sensor(accel) - pos_time

        # csv用の一行分のデータを作成
        data = list()
        data.extend(
            [pos_time.strftime('%Y-%m-%d'), pos_time.hour, pos_time.minute,
             pos_time.second, round(pos_time.microsecond / 1000, 1)])
        data.append(round(accel_dt.total_seconds() * 1000, 1))
        data.append(round(gyro_dt.total_seconds() * 1000, 1))
        data.append(delta_time.total_seconds())
        data.append(accel['x'])
        data.append(accel['y'])
        data.append(accel['z'])
        data.append(gyro['x'])
        data.append(gyro['y'])
        data.append(gyro['z'])
        data.append(pos['pos_x'])
        data.append(pos['pos_y'])
        data.append(pos['pos_z'])
        data.append(pos['rot_x'])
        data.append(pos['rot_y'])
        out_data.append(data)
        sys.stdout.write('\r' + '結合中: {}% ({}/{})'.format(int(pos_idx / (pos_last_idx - pos_idx_start) * 100),
                                                          pos_idx,
                                                          pos_last_idx - pos_idx_start))
        sys.stdout.flush()
        prev_time = pos_time

    # 出力するCSVデータの作成
    out_frame = pd.DataFrame(out_data)
    # ヘッダー(各数値の意味とか名前とかを表の一番上に入れとくやつ)設定
    out_frame.columns = ['date', 'hour', 'minute', 'sec', 'milli_sec',
                         'accel_dt[ms]', 'gyro_dt[ms]',
                         'delta_sec[s]',
                         'accel_x', 'accel_y', 'accel_z',
                         'gyro_x', 'gyro_y', 'gyro_z',
                         'pos_x', 'pos_y', 'pos_z',
                         'rot_x', 'rot_y']
    out_frame.to_csv(OUTPUT, index=False)
    out_frame.loc[:, 'pos_x':'rot_y'].to_csv(Y_OUT, index=False)
