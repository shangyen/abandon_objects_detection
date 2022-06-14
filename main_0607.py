import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
from model import*

def check_diff(background, foreimg):

    foreimg = cv2.cvtColor(foreimg, cv2.COLOR_BGR2GRAY)
    foreimg = cv2.GaussianBlur(foreimg, (5, 5), 0)
    
    diff = cv2.absdiff(background, foreimg)
    ret, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel, iterations=2) 
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    largest_contour_area = 800
    for cnt in contours:
        if cv2.contourArea(cnt) > largest_contour_area:
            largest_contour_area = cv2.contourArea(cnt)
            largest_contour = cnt

    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    img_output = np.zeros((680, 480, 3), np.uint8)  # pixel(y,x)
    img_output.fill(255)

    cv2.drawContours(img_output, [approx], 0, [0, 255, 0])

    cv2.namedWindow("check_diff", cv2.WINDOW_NORMAL)
    cv2.imshow("check_diff", img_output)
    

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


if __name__ == '__main__':

    print("Testing_Human_Detection_Case")

    # Setting camera
    URL = 0
    ipcam = ipcamCapture(URL)
    ipcam.start()
    time.sleep(1)

    human_detect = YOLOv4_human_detection()

    ROI_1 = np.array([[0.257, 0.228], [0.060, 0.471], [0.112, 0.726], [0.187, 0.853], [
        0.407, 0.865], [0.457, 0.693], [0.443, 0.427]], np.float32)
    ROI_2 = np.array([[0.65625, 0.4354166666666667], [0.9578125, 0.43333333333333335], [
                     0.9609375, 0.8333333333333334], [0.671875, 0.8333333333333334]], np.float32)
            
            
    # Need to prepare a background photo first
    background = cv2.imread("background.jpg")
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(background, (5, 5), 0)

    count = 0
    while True:
        frame = ipcam.getframe()
        height = frame.shape[0]
        width = frame.shape[1]
        another_frame = frame.copy()

        if count == 0:
            retentateTimerMap = np.zeros((height, width, 3), dtype="uint8")
            retentateTimerMapTmp = np.zeros(
                (height, width, 3), dtype="uint8")
            abandonObjectsLocation = []
            count += 1

        # Use for draw ROI polylines
        img_size = np.array([[width, 0], [0, height]])
        ROI_1_points = ROI_1.dot(img_size)
        ROI_1_points = ROI_1_points.reshape((-1, 1, 2))

        # click esc to destroy the analysis window
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows
            ipcam.stop()
            break

        # Human detecttion
        classes, confidences, boxes = human_detect.detect(frame)


        check_diff(background, another_frame)

        # Abandoned Objects detection _1
        start=time.time()
        frame, abandon, abandonObjectsLocation, retentateTimerMap, retentateTimerMapTmp = human_detect.abandonedObjects(
            ROI_1, frame, background, classes, confidences, boxes, retentateTimerMap, retentateTimerMapTmp)
        print(time.time()-start)

        if abandon == True:
            print("abandon_1:", abandon)
            print("abandonObjectsLocation_1:", abandonObjectsLocation)
        if abandon == False:
            print("abandon_1:", abandon)
            # print("retentateTimerMap_2:", retentateTimerMap_2)
            # print("abandonObjectsLocation_2:", abandonObjectsLocation_2)


        # Draw ROI area
        cv2.polylines(frame, pts=np.int32([ROI_1_points]), isClosed=True,
                      color=(0, 0, 255), thickness=3)
      

        # Draw human detection outputs frame
        human_detect.human_detection_outputs_frame(
            frame, classes, confidences, boxes)

        cv2.namedWindow("detect_result", cv2.WINDOW_NORMAL)
        cv2.imshow("detect_result", frame)

        # cv2.imwrite("background.jpg", frame)
