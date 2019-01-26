import os
import time
import tkinter.filedialog, tkinter.messagebox
import tkinter

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

    has_pred = False
    has_accel = False
    get_columns = ['pos_x', 'pos_y', 'pos_z']
    if 'pred_x' in data.columns:
        get_columns.extend(['pred_x', 'pred_y', 'pred_z'])
        has_pred = True
    if 'old_x' in data.columns:
        get_columns.extend(['old_x', 'old_y', 'old_z'])
        has_accel = True

    pos_data: pd.DataFrame = data.loc[:, get_columns]

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

    pos_width = right_pos - left_pos
    pos_height = down_pos - up_pos
    scale = WINDOW_SIZE / (pos_width if pos_width > pos_height else pos_height) / 2
    canvas.create_text(50, 15, text=u'教師データ', fill='gray', font=("Helvetica", 12, "bold"))
    canvas.create_text(50, 30, text=u'加速度積分', fill='cyan', font=("Helvetica", 12, "bold"))
    canvas.create_text(50, 45, text=u'NN予測', fill='pink', font=("Helvetica", 12, "bold"))
    for pos in pos_data.values:
        oval = canvas.create_oval(-CIRC_SIZE, -CIRC_SIZE, CIRC_SIZE, CIRC_SIZE, fill='gray')
        canvas.move(oval, pos[0]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[2]*scale)
        if has_pred:
            oval = canvas.create_oval(-CIRC_SIZE, -CIRC_SIZE, CIRC_SIZE, CIRC_SIZE, fill='pink')
            canvas.move(oval, pos[3]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[5]*scale)
            print(pos[0]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[2]*scale, pos[3]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[5]*scale)
            print('x誤差: {:.1f}cm'.format((pos[3] - pos[0])*100) + ',  z誤差: {:.1f}cm'.format((pos[5] - pos[3])*100))
            print('\n')
        if has_accel:
            if has_pred:
                oval = canvas.create_oval(-CIRC_SIZE, -CIRC_SIZE, CIRC_SIZE, CIRC_SIZE, fill='cyan')
                canvas.move(oval, pos[6]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[8]*scale)
            else:
                oval = canvas.create_oval(-CIRC_SIZE, -CIRC_SIZE, CIRC_SIZE, CIRC_SIZE, fill='cyan')
                canvas.move(oval, pos[3]*scale + WINDOW_SIZE/2, WINDOW_SIZE/2 - pos[5]*scale)
        time.sleep(0.045)
        root.update()
    # root.mainloop()
