import math

import pandas as pd
import sys
import datetime

OUTPUT = 'merged.csv'


def get_time_sensor(column):
    return get_time(column, '%Y/%m/%d')


def get_time_pos(column):
    return get_time(column, '%Y-%m-%d')


def get_time(column, date_format):
    tmp_date = datetime.datetime.strptime(column['date'], date_format)
    time = datetime.datetime(tmp_date.year, tmp_date.month, tmp_date.day,
                             int(column['hour']), int(column['min']),
                             int(math.modf(float(column['sec']))[1]),
                             int(math.modf(float(column['sec']))[0] * 1000000))
    return time


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python merge.py sensor.csv position.csv')
        exit(-1)

    # センサデータ読み込み & 各センサごとに分割
    sensor_input = pd.read_csv(filepath_or_buffer=sys.argv[1], encoding="utf-8")
    acceleration_data = sensor_input[sensor_input["sensor"] == 'Accel']
    gyro_data = sensor_input[sensor_input["sensor"] == 'Gyro']

    # 位置データ読み込み
    position_data = pd.read_csv(filepath_or_buffer=sys.argv[2], encoding="utf-8")

    start_time = get_time(position_data.iloc[0], "%Y-%m-%d")

    pos_last_idx = position_data.last_valid_index()
    # 各センサの記録開始時間
    accel_start = get_time_sensor(acceleration_data.iloc[0])
    gyro_start = get_time_sensor(gyro_data.iloc[0])

    # 記録開始が遅い方のセンサに合わせる
    slower_sensor_time = accel_start if accel_start > gyro_start else gyro_start
    pos_idx_start = 0  # 出力する最初の位置データ
    if slower_sensor_time > start_time:
        # センサデータの記録開始が遅い場合は
        # その記録が開始された時刻の位置データまでインデックスをずらす
        while get_time_pos(position_data.iloc[pos_idx_start]) < start_time:
            pos_idx_start = pos_idx_start + 1

    out_data = []
    for pos_idx in range(pos_idx_start, pos_last_idx):

        pos = position_data.iloc[pos_idx]
        pos_time = get_time_pos(pos)

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
             pos_time.second, round(pos_time.microsecond/1000, 1)])
        data.append(round(accel_dt.total_seconds()*1000, 1))
        data.append(round(gyro_dt.total_seconds()*1000, 1))
        data.append(pos['pos_x'])
        data.append(pos['pos_y'])
        data.append(pos['pos_z'])
        data.append(pos['rot_x'])
        data.append(pos['rot_y'])
        data.append(accel['x'])
        data.append(accel['y'])
        data.append(accel['z'])
        data.append(gyro['x'])
        data.append(gyro['y'])
        data.append(gyro['z'])
        out_data.append(data)
        sys.stdout.write('\r' + '結合中: {}% ({}/{})'.format(int(pos_idx/(pos_last_idx-pos_idx_start) * 100),
                                                             pos_idx,
                                                             pos_last_idx-pos_idx_start))
        sys.stdout.flush()

    # 出力するCSVデータの作成
    out_frame = pd.DataFrame(out_data)
    # ヘッダー(各数値の意味とか名前とかを表の一番上に入れとくやつ)設定
    out_frame.columns = ['date', 'hour', 'minute', 'sec', 'milli_sec',
                         'accel_dt[ms]', 'gyro_dt[ms]',
                         'pos_x', 'pos_y', 'pos_z',
                         'rot_x', 'rot_y',
                         'accel_x', 'accel_y', 'accel_z',
                         'gyro_x', 'gyro_y', 'gyro_z']
    out_frame.to_csv(OUTPUT, index=False)
