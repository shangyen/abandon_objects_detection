
# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(
    description='This script is used to demonstrate OpenPose human pose estimation network '
    'from https://github.com/CMU-Perceptual-Computing-Lab/openpose project using OpenCV. '
    'The sample and model are simplified and could be used for a single person on the frame.')

parser.add_argument('--input', default='C:\\Users\\user\\Desktop\\NADI_OpenCV\\'
                    'human_fall_detection\\openpose-opencv-master\\openpose-opencv-master\\'
                    'test.mp4', help='Path to video. Skip to capture frames from camera')

parser.add_argument('--ouput', default='C:\\Users\\user\\Desktop\\NADI_OpenCV\\'
                    'human_fall_detection\\openpose-opencv-master\\openpose-opencv-master\\'
                    'outpose.avi', help='Path to output video')

parser.add_argument('--proto', default='C:\\Users\\user\\Desktop\\NADI_OpenCV\\'
                    'human_fall_detection\\openpose-opencv-master\\openpose-opencv-master\\'
                    'pose_deploy.prototxt', help='Path to .prototxt')

parser.add_argument('--model', default='C:\\Users\\user\\Desktop\\NADI_OpenCV\\'
                    'human_fall_detection\\openpose-opencv-master\\openpose-opencv-master\\'
                    'pose_iter_440000.caffemodel', help='Path to .caffemodel')

parser.add_argument('--dataset', default='COCO', help='Specify what kind of model was trained. '
                    'It could be (COCO, MPI) depends on dataset.')

parser.add_argument('--thr', default=0.1, type=float,
                    help='Threshold value for pose parts heat map')

parser.add_argument('--width', default=368, type=int,
                    help='Resize input to specific width.')

parser.add_argument('--height', default=368, type=int,
                    help='Resize input to specific height.')

args = parser.parse_args()

if args.dataset == 'COCO':
    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder",
                                         "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], [
        "RKnee", "RAnkle"], ["Neck", "LHip"],
        ["LHip", "LKnee"], ["LKnee", "LAnkle"], [
            "Neck", "Nose"], ["Nose", "REye"],
        ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
else:
    assert(args.dataset == 'MPI')
    BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                  "Background": 15}

    POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["Neck", "LShoulder"], [
                      "LShoulder", "LElbow"],
                  ["LElbow", "LWrist"], ["Neck", "Chest"], [
                      "Chest", "RHip"], ["RHip", "RKnee"],
                  ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [
              0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromCaffe(args.proto, args.model)

#run in ceuda
net.setPrferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


cap = cv.VideoCapture(args.input if args.input else 0)
# frame quantity
frameNum = cap.get(cv.CAP_PROP_FRAME_COUNT)

# vedio writer
# fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
fourcc = cv.VideoWriter_fourcc(* 'XVID')  # 編碼格式 MPEG

# 保存size必須和輸出size設定一致，否則無法寫入保存檔案
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (w, h)
poseout = cv.VideoWriter(args.ouput, fourcc, 20.0, size)

# create a black use numpy
poseFrame = np.zeros((h, w, 3), np.uint8)

# fill the image with black
poseFrame.fill(0)

# cv.namedWindow('OpenPose')

print('Start...')
print('total{}frames.'.format(frameNum))

start_time = time.time()

j = 0
while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        # cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    # frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    # frame = cv.resize(frame, (inWidth, inHeight), interpolation=cv.INTER_AREA)

    inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                               (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)

    out = net.forward()

    # assert(len(BODY_PARTS) == out.shape[1])

    # reset
    points = []
    poseFrame.fill(0)
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.

        _, conf, _, point = cv.minMaxLoc(heatMap)

        x = (frameWidth * point[0]) / out.shape[3]  # width
        y = (frameHeight * point[1]) / out.shape[2]  # height

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)
    # *************** points ***********
    # print(points)

    i = 0
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        # assert(partFrom in BODY_PARTS)
        # assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(poseFrame, points[idFrom], points[idTo], colors[i], 3)
            cv.ellipse(poseFrame, points[idFrom], (3, 3),
                       0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(poseFrame, points[idTo], (3, 3),
                       0, 0, 360, (0, 0, 255), cv.FILLED)
        i += 1

    # t, _ = net.getPerfProfile()
    # freq = cv.getTickFrequency() / 1000
    # cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    # neck = points[BODY_PARTS['Neck']]
    # left_wrist = points[BODY_PARTS['LWrist']]
    # right_wrist = points[BODY_PARTS['RWrist']]
    # print(neck, left_wrist, right_wrist)
    # if neck and left_wrist and right_wrist and left_wrist[1] < neck[1] and right_wrist[1] < neck[1]:
    #     cv.putText(frame, 'HANDS UP!', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # cv.imshow('OpenPose', poseFrame)
    poseout.write(poseFrame)
    # cv.imwrite('outpose/pose_%d.jpg'%(j), frame)
    j += 1
    if j % 20 == 0:
        # 记录时间
        end_time = time.time()
        print('已处理{}/{}帧图像， 用时{:.4f}s， 平均每帧用时{:.4f}s'.format(j,
              int(frameNum), end_time - start_time, (end_time-start_time)/j))

# 计时结束
end_time = time.time()
print('共处理{}帧图像，用时{:.4f}s'.format(j, end_time - start_time))

cap.release()
poseout.release()
cv.destroyAllWindows()
