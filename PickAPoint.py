import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import cv2 as cv
import os


def getcoordinate(event):
    print(event.xdata, event.ydata)
    info = [event.xdata, event.ydata]
    imgcoordinate.append(info)
    draw_mark(imgcoordinate)
    # ----------
    return info


def getimgpoints(name):
    img = mpimg.imread(name)
    fig, ax = plt.subplots()
    plt.imshow(img)
    fig.canvas.mpl_connect('key_press_event', getcoordinate)
    plt.show()


def draw_mark(coor: list):
    img = cv.imread(img_name_formal)
    radious = 50
    color = (0, 0, 255)
    for i in coor:
        i = np.array(i, dtype=np.float64)
        coordinate = (int(i[0]), int(i[1]))
        cv.circle(img, coordinate, radious, color, -1)
    w, h = img.shape[:2]
    w = int(w/10)
    h = int(h/10)
    reimg = cv.resize(img, (h, w))
    #fig, ax = plt.subplots()

    # plt.imshow(img[:,:,::-1])
    # plt.show()
    cv.moveWindow(img_name, 500, 10)
    cv.imshow(img_name, reimg)

    cv.waitKey(0)


def save(imgcoordinate: list, num: int):
    input = {}
    savecoorx = []
    savecoory = []
    namex = str(num)+'_Xcoor'
    namey = str(num)+'_Ycoor'
    filename = str(num)+'_img_coordinate.xlsx'
    for data in range(len(imgcoordinate)):
        xdata = imgcoordinate[data][0]
        ydata = imgcoordinate[data][1]
        savecoorx.append(xdata)
        savecoory.append(ydata)
    input[namex] = savecoorx
    input[namey] = savecoory
    tosave = pd.DataFrame(input)
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    tosave.to_excel(writer)
    writer.save()


if __name__ == '__main__':
    count = 1
    imgcoordinate = []  # store the coordinate
    path = input('input your path here, which contain all the photos:')
    list_doc = os.listdir(path)
    count = 1
    for i in list_doc:
        img_name = i
        img_name_formal = path+'/'+img_name
        print('-------------' + img_name + '-------------')
        getimgpoints(path + '/' + img_name)
        save(imgcoordinate, count)
        print('-------------'+img_name+'-------------')
        count += 1
        imgcoordinate.clear()
