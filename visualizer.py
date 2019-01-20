import os
import time
import tkinter, tkinter.filedialog, tkinter.messagebox

import pandas as pd

WINDOW_SIZE = 512


def convert2list(pos_df: pd.DataFrame, sen_df: pd.DataFrame):
    return


# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()
fTyp = [("", "merge*.csv")]
iDir = os.path.abspath(os.path.dirname(__file__))
file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

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

    data = pd.read_csv(file)
    pos_data: pd.DataFrame = data.loc[:, 'pos_x':'pos_z']

    print(pos_data)
    left_pos = min(pos_data.loc[:, 'pos_x'].values)
    right_pos = max(pos_data.loc[:, 'pos_x'].values)
    down_pos = max(pos_data.loc[:, 'pos_z'].values)
    up_pos = min(pos_data.loc[:, 'pos_z'].values)

    pos_width = right_pos - left_pos
    pos_height = down_pos - up_pos
    scale = WINDOW_SIZE / (pos_width if pos_width > pos_height else pos_height) / 2
    for pos in pos_data.values:
        oval = canvas.create_oval(-10, -10, 10, 10, fill='pink')
        canvas.move(oval, pos[0]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[2]*scale)
        print(pos[0]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[2]*scale)
        time.sleep(0.045)
        root.update()
