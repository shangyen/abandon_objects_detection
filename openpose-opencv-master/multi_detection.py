import tkinter as tk
from tkinter import filedialog
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


num_points = 18

keypointsMapping = ['Nose', 'Neck',
                    'R-Sho', 'R-Elb', 'R-Wr',
                    'L-Sho', 'L-Elb', 'L-Wr',
                    'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank',
                    'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
              [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
              [2, 17], [5, 16]]

# BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
#               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

# POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#               ["Neck", "RHip"], ["RHip", "RKnee"], [
#     "RKnee", "RAnkle"], ["Neck", "LHip"],
#     ["LHip", "LKnee"], ["LKnee", "LAnkle"], [
#     "Neck", "Nose"], ["Nose", "REye"],
#     ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
          [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
          [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
          [37, 38], [45, 46]]

colors = [[0, 100, 255], [0, 100, 255],   [0, 255, 255],
          [0, 100, 255], [0, 255, 255],   [0, 100, 255],
          [0, 255, 0],   [255, 200, 100], [255, 0, 255],
          [0, 255, 0],   [255, 200, 100], [255, 0, 255],
          [0, 0, 255],   [255, 0, 0],     [200, 200, 0],
          [255, 0, 0],   [200, 200, 0],   [0, 0, 0]]

# dnn 加载模型

filename = "C:\\Users\\user\\Desktop\\NADI_OpenCV\\human_fall_detection\\openpose-opencv-master\\openpose-opencv-master"
protoFile = filename + "\\pose_deploy.prototxt"
weightsFile = filename + "\\pose_iter_440000.caffemodel"

start = time.time()

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
print("[INFO]Time Taken in Model Loading: {}".format(time.time() - start))

# test img
root = tk.Tk()
root.withdraw()

print("Which image you wanna do pose estimation?")
file_path = filedialog.askopenfilename(parent=root)

img = cv2.imread(file_path)
img_width, img_height = img.shape[1], img.shape[0]

# 根据长宽比，固定网路输入的 height，计算网络输入的 width.
net_height = 368
net_width = int((net_height/img_height)*img_width)

start = time.time()
in_blob = cv2.dnn.blobFromImage(
    img,
    1.0 / 255,
    (net_width, net_height),
    (0, 0, 0),
    swapRB=False,
    crop=False)

net.setInput(in_blob)
output = net.forward()
print("[INFO]Time Taken in Forward pass: {}".format(time.time() - start))


def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
    mapMask = np.uint8(mapSmooth > threshold)

    keypoints = []
    # 1. 找出对应于关键点的所有区域的轮廓(contours)
    contours, hierarchy = cv2.findContours(
        mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for each blob find the maxima
    # 对于每个关键点轮廓区域，找到最大值.
    for cnt in contours:
        # 2. 创建关键点的 mask；
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        # 3. 提取关键点区域的 probMap
        maskedProbMap = mapSmooth * blobMask
        # 4. 提取关键点区域的局部最大值.
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints
