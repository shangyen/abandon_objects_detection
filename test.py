from ast import Dict
import cv2
import os
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# readpath = "C:/Users/Maurice_Hsu/Desktop/labeling/123"
# savepath = "C:/Users/Maurice_Hsu/Desktop/labeling/car_license_plate_number/"

root = tk.Tk()
root.withdraw()

print("select where you want to read:")
readpath = filedialog.askdirectory()

print("select where you want to save:")
savepath = filedialog.askdirectory()

files= os.listdir(readpath)

my_dict = []

for file in files:
    if not os.path.isdir(file):
        print("file name: ", file)
        img = cv2.imread(readpath +"/"+ file)
        # imgplot = plt.imshow(img)
        # plt.show()
        cv2.namedWindow("original",0)
        cv2.imshow('original', img)

        選擇ROI
        roi = cv2.selectROI(windowName="original", img=img, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi

        # 顯示ROI並儲存圖片
        if roi != (0, 0, 0, 0):
            crop = img[y:y+h, x:x+w]
            cv2.imshow('crop', crop)

            plate_number = str(input())
            file = str(file)
            my_dict.append([file,plate_number])

        
            textlen = len(file)
            filename = str(file)
            filename = filename[0:textlen-4]
            cv2.imwrite(savepath +"/"+ filename + ".jpg", crop)
            print('Saved!')


b = dict(my_dict)
with open(savepath+'/data.json', 'w') as fp:
    json.dump(b, fp)

# 退出

cv2.waitKey(0)
cv2.destroyAllWindows()