import math

import pandas as pd
import sys
import datetime

OUTPUT = 'merged.csv'


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

    first_accel = acceleration_data.iloc[0]
    accel_start = get_time(first_accel, "%Y/%m/%d")

    first_gyro = gyro_data.iloc[0]
    gyro_start = get_time(first_gyro, "%Y/%m/%d")

    first_pos = position_data.iloc[0]
    pos_start = get_time(first_pos, "%Y-%m-%d")

    print('accel start: ' + str(accel_start))
    print('gyro start: ' + str(gyro_start))
    print('pos start: ' + str(pos_start))

'''
    rec_start_time = max([accel_start, gyro_start, pos_start])
    print(rec_start_time)

    accel_idx = 0
    while get_time(acceleration_data.iloc[accel_idx], "%Y/%m/%d") < rec_start_time:
        accel_idx = accel_idx + 1

    gyro_idx = 0
    while get_time(gyro_data.iloc[gyro_idx], "%Y/%m/%d") < rec_start_time:
        gyro_idx = gyro_idx + 1

    pos_idx = 0
    while get_time(position_data.iloc[pos_idx], "%Y-%m-%d") < rec_start_time:
        pos_idx = pos_idx + 1

    print(acceleration_data.iloc[accel_idx]['sec'])
    print(gyro_data.iloc[gyro_idx]['sec'])
    print(position_data.iloc[pos_idx]['sec'])
'''
    
