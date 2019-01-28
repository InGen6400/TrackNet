import math
import os
import time
import tkinter.filedialog, tkinter.messagebox
import tkinter
from typing import List, Any

import matplotlib.pyplot as plt

import pandas as pd

WINDOW_SIZE = 512


def convert2list(pos_df: pd.DataFrame, sen_df: pd.DataFrame):
    return


# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()
fTyp = [("", "*.csv")]
iDir = os.path.abspath(os.path.dirname(__file__))
file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

CIRC_SIZE = 15


class Data:
    def __init__(self, color):
        self.line_opt = color
        self.X = []
        self.Y = []

    def add(self, x, y):
        self.X.append(x)
        self.Y.append(y)


def draw_graph(*data_list):
    for data in data_list:
        if data is not None:
            line, = plt.plot(data.X, data.Y, color=data.line_opt, label="error")
            line.set_ydata(data.Y)
    plt.title(u"HoloLensとの誤差")
    plt.xlabel("t[s]")
    plt.ylabel("error[m]")
    plt.legend()
    plt.grid()
    plt.xlim([0, 12])
    plt.ylim([0, 8])
    plt.draw()
    plt.pause(0.05)
    plt.clf()


if __name__ == "__main__":
    root = tkinter.Tk()
    root.title(u"座標表示")
    root.geometry("512x512")

    canvas = tkinter.Canvas(root, width=WINDOW_SIZE, height=WINDOW_SIZE, bd=0)
    canvas.pack()
    canvas.place(x=0, y=0)

    # 軸
    canvas.create_line(0, WINDOW_SIZE/2, WINDOW_SIZE, WINDOW_SIZE/2, fill='red', arrow=tkinter.LAST)
    canvas.create_line(WINDOW_SIZE/2, 0, WINDOW_SIZE/2, WINDOW_SIZE, fill='green', arrow=tkinter.FIRST)

    root.update()

    csv_data = pd.read_csv(file)

    has_pred = False
    has_accel = False
    get_columns = ['dt[s]', 'pos_x', 'pos_y', 'pos_z']
    if 'pred_x' in csv_data.columns:
        get_columns.extend(['pred_x', 'pred_y', 'pred_z'])
        has_pred = True
    if 'old_x' in csv_data.columns:
        get_columns.extend(['old_x', 'old_y', 'old_z'])
        has_accel = True

    pos_data: pd.DataFrame = csv_data.loc[:, get_columns]

    print(pos_data)
    left_pos = min(pos_data.loc[:, 'pos_x'].values)
    right_pos = max(pos_data.loc[:, 'pos_x'].values)
    down_pos = max(pos_data.loc[:, 'pos_z'].values)
    up_pos = min(pos_data.loc[:, 'pos_z'].values)

    if has_pred:
        left_pos = min([left_pos, min(pos_data.loc[:, 'pred_x'].values)])
        right_pos = max([right_pos, max(pos_data.loc[:, 'pred_x'].values)])
        down_pos = max([down_pos, max(pos_data.loc[:, 'pred_z'].values)])
        up_pos = min([up_pos, min(pos_data.loc[:, 'pred_z'].values)])
    if has_accel:
        left_pos = min([left_pos, min(pos_data.loc[:, 'old_x'].values)])
        right_pos = max([right_pos, max(pos_data.loc[:, 'old_x'].values)])
        down_pos = max([down_pos, max(pos_data.loc[:, 'old_z'].values)])
        up_pos = min([up_pos, min(pos_data.loc[:, 'old_z'].values)])

    pred_data = Data("#DDA0DD")
    accel_data = Data("#00FFFF")

    pos_width = right_pos - left_pos
    pos_height = down_pos - up_pos
    scale = WINDOW_SIZE / (pos_width if pos_width > pos_height else pos_height) / 2
    canvas.create_text(50, 15, text=u'教師データ', fill='gray', font=("Helvetica", 12, "bold"))
    canvas.create_text(50, 30, text=u'加速度積分', fill='cyan', font=("Helvetica", 12, "bold"))
    canvas.create_text(50, 45, text=u'NN予測', fill='pink', font=("Helvetica", 12, "bold"))
    for pos in pos_data.values:
        oval = canvas.create_oval(-CIRC_SIZE, -CIRC_SIZE, CIRC_SIZE, CIRC_SIZE, fill='gray')
        canvas.move(oval, pos[1]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[3]*scale)
        time.sleep(0.005)
        root.update()

    for pos in pos_data.values:
        if has_accel:
            if has_pred:
                oval = canvas.create_oval(-CIRC_SIZE, -CIRC_SIZE, CIRC_SIZE, CIRC_SIZE, fill='cyan')
                canvas.move(oval, pos[7]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[9]*scale)
                delta = (pos[1] - pos[7])**2 + (pos[3] - pos[9])**2
                accel_data.add(pos[0], math.sqrt(delta))
            else:
                oval = canvas.create_oval(-CIRC_SIZE, -CIRC_SIZE, CIRC_SIZE, CIRC_SIZE, fill='cyan')
                canvas.move(oval, pos[4]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[6]*scale)
                delta = (pos[1] - pos[4])**2 + (pos[3] - pos[6])**2
                accel_data.add(pos[0], math.sqrt(delta))
        if has_pred:
            oval = canvas.create_oval(-CIRC_SIZE, -CIRC_SIZE, CIRC_SIZE, CIRC_SIZE, fill='pink')
            canvas.move(oval, pos[4]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[6]*scale)
            delta = (pos[1] - pos[4])**2 + (pos[3] - pos[6])**2
            pred_data.add(pos[0], math.sqrt(delta))
            print(pos[1]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[3]*scale, pos[4]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[6]*scale)
            print('x誤差: {:.1f}cm'.format((pos[4] - pos[1])*100) + ',  z誤差: {:.1f}cm'.format((pos[6] - pos[4])*100))
            print('\n')
        draw_graph(accel_data, pred_data)
        root.update()


    root.mainloop()
