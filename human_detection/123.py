import cv2
import numpy as np
import threading
import time


class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(URL)

    def start(self):
        print("ipcam started!")
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        self.isstop = True
        print("ipcam stopped")

    def getframe(self):
        return self.Frame.copy()

    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        self.capture.release


background = cv2.imread("background.jpg")
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background = cv2.GaussianBlur(background, (5, 5), 0)

URL = 0
ipcam = ipcamCapture(URL)
ipcam.start()
time.sleep(1)

ROI = np.array([[0.257, 0.228], [0.060, 0.471], [0.112, 0.726], [0.187, 0.853], [
                0.407, 0.865], [0.457, 0.693], [0.443, 0.427]], np.float32)


while True:
    frame = ipcam.getframe()
    height = frame.shape[0]
    width = frame.shape[1]

    # Use for draw ROI polylines
    img_size = np.array([[width, 0], [0, height]])
    ROI_points = ROI.dot(img_size)
    ROI_points = ROI_points.reshape((-1, 1, 2))

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows
        ipcam.stop()
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # 計算目前影格與平均影像的差異值
    diff = cv2.absdiff(background, frame)

    # 篩選出變動程度大於門檻值的區域
    ret, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # # 使用型態轉換函數去除雜訊
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 產生等高線
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    largest_contour_area = 100
    for cnt in contours:
        if cv2.contourArea(cnt) > largest_contour_area:
            largest_contour_area = cv2.contourArea(cnt)
            largest_contour = cnt

    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    img_output = np.zeros((680, 480, 3), np.uint8)  # pixel(y,x)
    img_output.fill(255)

    final_2 = cv2.drawContours(img_output, [largest_contour], 0, [0, 255, 0])

    # # Draw ROI area
    # cv2.polylines(img_output, pts=np.int32([ROI_points]), isClosed=True,
    #               color=(0, 0, 255), thickness=3)

    cv2.namedWindow("result_2", cv2.WINDOW_NORMAL)
    cv2.imshow("result_2", img_output)
