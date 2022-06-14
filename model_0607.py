import traceback
import cv2
import subprocess
import json
import numpy as np
from threading import RLock

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)


def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (
        nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = (line.strip() for line in lines if line.strip() != '')

    return [{k: v for k, v in zip(keys, line.split(', '))} for line in lines]


class YOLOv4_human_detection(object):

    def __init__(self):
        """ Method called when object of this class is created. """
        self.net = None
        self.model_lock = RLock()
        self.frame_lock = RLock()
        self.matadata_lock = RLock()
        self.invasionMask_lock = RLock()
        self.retentate_lock = RLock()
        self.initialize_network()

    def initialize_network(self):
        self.gpu_devid = 0
        gpu_top_devid = 0
        gpu_dev_info = get_gpu_info()
        # print(gpu_dev_info)
        if len(gpu_dev_info) > 1:
            gpu_top_devid = gpu_dev_info[0]['index']
            gpu_top_memory = gpu_dev_info[0]['memory.free']
            for i in range(len(gpu_dev_info)):
                gpu_tmp_memory = gpu_dev_info[i]['memory.free']
                if int(gpu_tmp_memory) > int(gpu_top_memory):
                    gpu_top_devid = gpu_dev_info[i]['index']
                    gpu_top_memory = gpu_dev_info[i]['memory.free']
        self.gpu_devid = gpu_top_devid
        """ Method to initialize and load the model. """
        self.net = cv2.dnn_DetectionModel(
            f"./data/yolov4.cfg", f"./data/yolov4.weights")
        cv2.cuda.setDevice(int(gpu_top_devid))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.net.setInputSize(416, 416)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)
        with open(f"./data/coco.names", 'rt') as f:
            self.names = f.read().rstrip('\n').split('\n')

    def detect(self, frame, confThreshold=0.5, nmsThreshold=0.5):
        self.model_lock.acquire()
        try:
            classes, confidences, boxes = self.net.detect(
                frame, confThreshold, nmsThreshold)
        finally:
            self.model_lock.release()
            return classes, confidences, boxes

    def get_gpuid(self):
        return self.gpu_devid

    def human_detection_outputs_frame(self, img, classes, confidences, boxes):
        self.frame_lock.acquire()
        try:
            if len(classes) != 0:
                for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                    if self.names[classId] == 'person':
                        box = np.array(box)
                        x1y1 = (np.array(box[0:2])).astype(np.int32)
                        x2y2 = (
                            np.array([box[0] + box[2], box[1] + box[3]])).astype(np.int32)
                        img = cv2.rectangle(img, tuple(
                            x1y1), tuple(x2y2), (0, 255, 255), 1)
                        img = cv2.putText(
                            img, 'Person', (x1y1[0], x1y1[1]-3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
        finally:
            self.frame_lock.release()
            return img

    def outputs_matadata(self, img, classes, confidences, boxes):
        self.matadata_lock.acquire()
        try:
            analysis_result = []
            if len(classes) != 0:
                w = img.shape[1]
                h = img.shape[0]
                for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                    if self.names[classId] == 'person':
                        box = np.array(box)
                        x1y1 = [(box[0]/w), (box[1]/h)]
                        wh = [(box[2]/w), (box[3]/h)]
                        analysis_result.append([x1y1, wh])
        finally:
            self.matadata_lock.release()
            return analysis_result

    def invasionMask(self, ROI, frame, classes, confidences, boxes):
        self.invasionMask_lock.acquire()
        try:
            width = frame.shape[1]
            height = frame.shape[0]
            mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype="uint8")
            invasion_alarm = False
            analysis_result = []
            # for i in range(len(ROI)):
            # if ROI[i] != "":
            # ROI_json = json.loads(ROI[i])
            cordinates = np.array(ROI) * np.array([width, height])
            cordinates = cordinates.astype(np.int32)
            mask = cv2.fillPoly(mask, [cordinates], (0, 0, 255))
            if len(classes) > 0 and len(confidences) > 0:
                for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                    if self.names[classId] == 'person':
                        box = np.array(box)
                        x1y1 = [(box[0]/width), (box[1]/height)]
                        wh = [(box[2]/width), (box[3]/height)]
                        if mask[int(box[1] + box[3]) - 1, (box[0] + int(box[2] / 2)) - 1, :][2] == 255:
                            analysis_result.append([x1y1, wh])
                            invasion_alarm = True
        finally:
            self.invasionMask_lock.release()
            return invasion_alarm, analysis_result

    def retentateROI(self, ROI, frame):
        width = frame.shape[1]
        height = frame.shape[0]
        mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype="uint8")
        # for i in range(len(ROI)):
        # if ROI[i] != "":
        #     ROI_json = json.loads(ROI[i])
        cordinates = np.array(ROI) * np.array([width, height])
        cordinates = cordinates.astype(np.int32)
        mask = cv2.fillPoly(mask, [cordinates], (0, 0, 255))
        # cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        # cv2.imshow("ROI", mask)
        return mask

    def abandonedObjects(self, ROI, frame, background, classes, confidences, boxes, retentateTimerMap, retentateTimerMapTmp):
        # 使用這支 function 時，要自己另外給 retentateTimerMap and retentateTimerMapTmp (詳情看 main.py lines 57,58)
        ### 用途是為了累計遺留物偵測所計算的值 (超過閥值才會認定是滯留物)
        # example:
        ### retentateTimerMap = np.zeros((height, width, 3), dtype="uint8")
        ### retentateTimerMapTmp = np.zeros((height, width, 3), dtype="uint8")

        self.retentate_lock.acquire()
        try:
            flag = 1
            abandon = False
            abandonObjectsLocation = []
            width = frame.shape[1]
            height = frame.shape[0]
            foreground = frame.copy()
            foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
            foreground = cv2.GaussianBlur(foreground, (5, 5), 0)

            # 計算目前影格與平均影像的差異值
            diff = cv2.absdiff(background, foreground)

            # 篩選出變動程度大於門檻值的區域
            ret, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

            # # 使用型態轉換函數去除雜訊
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 產生等高線
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)



            if len(contours) == 0:
                abandon = False
                return frame, abandon, abandonObjectsLocation,retentateTimerMap,retentateTimerMapTmp
            else:
                c_check = False
                
                for c in contours:

                    area = cv2.contourArea(c)
                    if area > 500:
                        largest_contour = c
                        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                        (x, y, w, h) = cv2.boundingRect(approx)
                        
                        RoiMap = self.retentateROI(ROI, frame)
                        if RoiMap[y + int(h/2) - 1, x + int(w/2) - 1, :][2] == 255 or RoiMap[y + h - 1, x + int(w/2) - 1, :][2] == 255:
                            retentateTimerMapTmp[y:y + h, x:x + w, 2] = 255
                            # 如果輪廓中心點有再ROI裡面，則會累加 retentateTimerMap 的值
                            retentateTimerMap[y:y + h, x:x + w, 2] += 1                        
                            c_check = True

                # 如果都沒有抓到輪廓 = 不進行不進行累加 retentateTimerMap 值
                if c_check == False:
                    retentateTimerMapTmp=np.zeros((height, width, 3), dtype="uint8")
                    retentateTimerMap=np.zeros((height, width, 3), dtype="uint8")
                    return frame, abandon, abandonObjectsLocation,retentateTimerMap,retentateTimerMapTmp


                else:
                    retentateTimerMap[retentateTimerMapTmp != 255] //= 2
                    retentateTimerMap.astype(np.uint8)

                    # 當 retentateTimerMap 的值 累積到一定的值後，才會標記為是滯留物
                    if (retentateTimerMap >= 100).any():
                        tmpMap = retentateTimerMap.copy()
                        tmpMap.astype(np.uint8)

                        ret, thresh2 = cv2.threshold(
                            tmpMap[:, :, 2], 99, 255, cv2.THRESH_BINARY)

                        contours, hierarchy = cv2.findContours(
                            thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        for x in contours:
                            (x_1, y_1, w_1, h_1) = cv2.boundingRect(x)
                            if len(classes) != 0:
                                for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                                    if (box[0] <= (x_1+w_1) or (box[0]+box[2]) >= x_1) or (box[1] <= (y_1+h_1) or (box[1]+box[3]) >= y_1):
                                        if self.names[classId] == 'person':
                                            continue
                                        else:
                                            cv2.rectangle(
                                                frame, (x_1, y_1), (x_1 + w_1, y_1 + h_1), (255, 0, 0), 2)
                                            if [[x_1 / width, y_1 / height], w_1 / width, h_1 / height] not in abandonObjectsLocation:
                                                abandonObjectsLocation.append(
                                                    [[x_1 / width, y_1 / height], w_1 / width, h_1 / height])
                            else:
                                cv2.rectangle(
                                    frame, (x_1, y_1), (x_1 + w_1, y_1 + h_1), (255, 0, 0), 2)
                                if [[x_1 / width, y_1 / height], w_1 / width, h_1 / height] not in abandonObjectsLocation:
                                    abandonObjectsLocation.append(
                                        [[x_1 / width, y_1 / height], w_1 / width, h_1 / height])

                    if len(abandonObjectsLocation) != 0:
                        abandon = True
                    else:
                        abandon = False

        except:
            traceback.print_exc()
        finally:
            self.retentate_lock.release()
            return frame, abandon, abandonObjectsLocation,retentateTimerMap,retentateTimerMapTmp
