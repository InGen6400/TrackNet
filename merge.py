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

    sensor_input = pd.read_csv(filepath_or_buffer=sys.argv[1], encoding="utf-8")
    acceleration_data = sensor_input[sensor_input["sensor"] == 'Accel']
    gyro_data = sensor_input[sensor_input["sensor"] == 'Gyro']
    position_data = pd.read_csv(filepath_or_buffer=sys.argv[2], encoding="utf-8")

    start_time = get_time(position_data.iloc[0], "%Y-%m-%d")

    print('pos start: ' + str(start_time))

    accel_idx = 0
    gyro_idx = 0
    # 加速度センサのデータのほうが早くスタートしているとき
    # 加速度センサのインデックスを位置データの時刻に合わせる
    while get_time_sensor(acceleration_data.iloc[accel_idx]) < start_time:
        accel_idx = accel_idx + 1

    # ジャイロセンサのデータのほうが早くスタートしているとき
    # ジャイロセンサのインデックスを位置データの時刻に合わせる
    while get_time_sensor(gyro_data.iloc[gyro_idx]) < start_time:
        gyro_idx = gyro_idx + 1

    pos_idx = 0
    while get_time_pos(position_data.iloc[pos_idx]) < get_time_sensor(acceleration_data.iloc[accel_idx]):
        pos_idx = pos_idx + 1

    # 位置データ一歩優先
    if pos_idx - 1 >= 0:
        pos_idx = pos_idx - 1

    print(position_data.last_valid_index())
    out_data = []
    pos_last_idx = position_data.last_valid_index()
    accel_last_idx = acceleration_data.last_valid_index()
    gyro_last_idx = gyro_data.last_valid_index()

    while pos_idx <= pos_last_idx:
        pos = position_data.iloc[pos_idx]
        pos_time = get_time_pos(pos)
        # 一つ前のセンサデータと比較して近い方を選択
        dt1 = get_time_sensor(acceleration_data.iloc[accel_idx - 1]) - pos_time
        dt2 = get_time_sensor(acceleration_data.iloc[accel_idx]) - pos_time
        if abs(dt1) < abs(dt2):
            accel = acceleration_data.iloc[accel_idx - 1]
            accel_dt = dt1
        else:
            accel = acceleration_data.iloc[accel_idx]
            # 現在のデータのほうが近いならインデックスを進める
            accel_idx = accel_idx + 1
            accel_dt = dt2

        # 一つ前のセンサデータと比較して近い方を選択
        dt1 = get_time_sensor(gyro_data.iloc[gyro_idx - 1]) - pos_time
        dt2 = get_time_sensor(gyro_data.iloc[gyro_idx]) - pos_time
        if abs(dt1) < abs(dt2):
            gyro = gyro_data.iloc[gyro_idx - 1]
            gyro_dt = dt1
        else:
            gyro = gyro_data.iloc[gyro_idx]
            # 現在のデータのほうが近いならインデックスを進める
            gyro_idx = gyro_idx + 1
            gyro_dt = dt2

        data = list()
        data.extend(
            [pos_time.strftime('%Y-%m-%d'), pos_time.hour, pos_time.minute, pos_time.second, pos_time.microsecond])
        data.append(accel_dt.total_seconds())
        data.append(gyro_dt.total_seconds())
        data.append(pos['x'])
        data.append(pos['y'])
        data.append(pos['z'])
        data.append(accel['x'])
        data.append(accel['y'])
        data.append(accel['z'])
        data.append(gyro['x'])
        data.append(gyro['y'])
        data.append(gyro['z'])
        out_data.append(data)
        pos_idx = pos_idx + 1

    out_frame = pd.DataFrame(out_data)
    out_frame.columns = ['date', 'hour', 'minute', 'sec', 'micro_sec',
                         'accel_dt[sec]', 'gyro_dt[sec]',
                         'pos_x', 'pos_y', 'pos_z',
                         'accel_x', 'accel_y', 'accel_z',
                         'gyro_x', 'gyro_y', 'gyro_z']
    out_frame.to_csv(OUTPUT, index=False)
