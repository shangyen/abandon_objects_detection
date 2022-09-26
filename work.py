import cv2
import os
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()

print("select where you want to read:")
readpath = filedialog.askdirectory()

print("select where you want to save:")
savepath = filedialog.askdirectory()

files = os.listdir(readpath)

# 全域性變數
g_window_name = "img"  # 視窗名
g_window_wh = [800, 600]  # 視窗寬高

g_location_win = [0, 0]  # 相對於大圖，視窗在圖片中的位置
location_win = [0, 0]  # 滑鼠右鍵點選時，暫存g_location_win
g_location_click, g_location_release = [0, 0], [0, 0]  # 相對於視窗，滑鼠左鍵點選和釋放的位置

g_zoom, g_step = 1, 0.1  # 圖片縮放比例和縮放係數

my_dict = []
c = 1
# 矯正視窗在圖片中的位置
# img_wh:圖片的寬高, win_wh:視窗的寬高, win_xy:視窗在圖片的位置


def check_location(img_wh, win_wh, win_xy):
    for i in range(2):
        if win_xy[i] < 0:
            win_xy[i] = 0
        elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] > win_wh[i]:
            win_xy[i] = img_wh[i] - win_wh[i]
        elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] < win_wh[i]:
            win_xy[i] = 0
    # print(img_wh, win_wh, win_xy)


# 計算縮放倍數
# flag：滑鼠滾輪上移或下移的標識, step：縮放係數，滾輪每步縮放0.1, zoom：縮放倍數
def count_zoom(flag, step, zoom):
    if flag > 0:  # 滾輪上移
        zoom += step
        if zoom > 1 + step * 20:  # 最多能放大到3倍
            zoom = 1 + step * 20
    else:  # 滾輪下移
        zoom -= step
        if zoom < step:  # 最多隻能縮小到0.1倍
            zoom = step
    zoom = round(zoom, 2)  # 取2位有效數字
    return zoom


# OpenCV滑鼠事件
def mouse(event, x, y, flags, param):
    global g_location_click, g_location_release, g_image_show, g_image_zoom, g_location_win, location_win, g_zoom, c
    if event == cv2.EVENT_RBUTTONDOWN:  # 右鍵點選
        g_location_click = [x, y]  # 右鍵點選時，滑鼠相對於視窗的座標
        # 視窗相對於圖片的座標，不能寫成location_win = g_location_win
        location_win = [g_location_win[0], g_location_win[1]]
    # 按住右鍵拖曳
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):
        g_location_release = [x, y]  # 右鍵拖曳時，滑鼠相對於視窗的座標
        h1, w1 = g_image_zoom.shape[0:2]  # 縮放圖片的寬高
        w2, h2 = g_window_wh  # 視窗的寬高
        show_wh = [0, 0]  # 實際顯示圖片的寬高
        if w1 < w2 and h1 < h2:  # 圖片的寬高小於視窗寬高，無法移動
            show_wh = [w1, h1]
            g_location_win = [0, 0]
        elif w1 >= w2 and h1 < h2:  # 圖片的寬度大於視窗的寬度，可左右移動
            show_wh = [w2, h1]
            g_location_win[0] = location_win[0] + \
                g_location_click[0] - g_location_release[0]
        elif w1 < w2 and h1 >= h2:  # 圖片的高度大於視窗的高度，可上下移動
            show_wh = [w1, h2]
            g_location_win[1] = location_win[1] + \
                g_location_click[1] - g_location_release[1]
        else:  # 圖片的寬高大於視窗寬高，可左右上下移動
            show_wh = [w2, h2]
            g_location_win[0] = location_win[0] + \
                g_location_click[0] - g_location_release[0]
            g_location_win[1] = location_win[1] + \
                g_location_click[1] - g_location_release[1]
        check_location([w1, h1], [w2, h2], g_location_win)  # 矯正視窗在圖片中的位置
        g_image_show = g_image_zoom[g_location_win[1]:g_location_win[1] +
                                    show_wh[1], g_location_win[0]:g_location_win[0] + show_wh[0]]  # 實際顯示的圖片
    elif event == cv2.EVENT_MOUSEWHEEL:  # 滾輪
        z = g_zoom  # 縮放前的縮放倍數，用於計算縮放後窗口在圖片中的位置
        g_zoom = count_zoom(flags, g_step, g_zoom)  # 計算縮放倍數
        w1, h1 = [int(g_image_original.shape[1] * g_zoom),
                  int(g_image_original.shape[0] * g_zoom)]  # 縮放圖片的寬高
        w2, h2 = g_window_wh  # 視窗的寬高
        g_image_zoom = cv2.resize(
            g_image_original, (w1, h1), interpolation=cv2.INTER_AREA)  # 圖片縮放
        show_wh = [0, 0]  # 實際顯示圖片的寬高
        if w1 < w2 and h1 < h2:  # 縮放後，圖片寬高小於視窗寬高
            show_wh = [w1, h1]
            cv2.resizeWindow(g_window_name, w1, h1)
        elif w1 >= w2 and h1 < h2:  # 縮放後，圖片高度小於視窗高度
            show_wh = [w2, h1]
            cv2.resizeWindow(g_window_name, w2, h1)
        elif w1 < w2 and h1 >= h2:  # 縮放後，圖片寬度小於視窗寬度
            show_wh = [w1, h2]
            cv2.resizeWindow(g_window_name, w1, h2)
        else:  # 縮放後，圖片寬高大於視窗寬高
            show_wh = [w2, h2]
            cv2.resizeWindow(g_window_name, w2, h2)
        g_location_win = [int((g_location_win[0] + x) * g_zoom / z - x),
                          int((g_location_win[1] + y) * g_zoom / z - y)]  # 縮放後，視窗在圖片的位置
        check_location([w1, h1], [w2, h2], g_location_win)  # 矯正視窗在圖片中的位置
        # print(g_location_win, show_wh)
        g_image_show = g_image_zoom[g_location_win[1]:g_location_win[1] +
                                    show_wh[1], g_location_win[0]:g_location_win[0] + show_wh[0]]  # 實際的顯示圖片
    elif event == cv2.EVENT_LBUTTONDOWN:
        roi = cv2.selectROI(g_window_name, g_image_show,
                            showCrosshair=True, fromCenter=False)
        xx, yy, w, h = roi
        # 顯示ROI並儲存圖片
        if roi != (0, 0, 0, 0):
            crop = g_image_show[yy:yy+h, xx:xx+w]
            cv2.imshow('crop', crop)
            plate_number = str(input())
            filename = "nmuber_pic_"

            my_dict.append([filename + str(c), plate_number])
            cv2.imwrite(savepath + "/" + filename + str(c) + ".jpg", crop)
            c += 1
            print('Saved!')

    cv2.imshow(g_window_name, g_image_show)


# 主函式
if __name__ == "__main__":

    cv2.namedWindow(g_window_name, cv2.WINDOW_NORMAL)
    # 設定視窗大小，只有當圖片大於視窗時才能移動圖片
    cv2.resizeWindow(g_window_name, g_window_wh[0], g_window_wh[1])
    cv2.moveWindow(g_window_name, 700, 100)  # 設定視窗在電腦螢幕中的位置

    for file in files:
        if not os.path.isdir(file):
            print("file name: ", file)
            g_image_original = cv2.imread(readpath + "/" + file)
            g_image_zoom = g_image_original.copy()  # 縮放後的圖片
            g_image_show = g_image_original[g_location_win[1]:g_location_win[1] +
                                            g_window_wh[1], g_location_win[0]:g_location_win[0] + g_window_wh[0]]  # 實際顯示的圖片
            # 滑鼠事件的回撥函式
            cv2.setMouseCallback(g_window_name, mouse)
            cv2.waitKey(0)

    # b = dict(my_dict)
    # with open(savepath+'/data.json', 'w') as fp:
    #     json.dump(b, fp)

    cv2.waitKey(0)  # 不可缺少，用於重新整理圖片，等待滑鼠操作
    cv2.destroyAllWindows()
