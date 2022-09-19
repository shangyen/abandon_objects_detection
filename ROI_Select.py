import cv2
import imutils
import numpy as np
import joblib

pts = []  # 用於存放點
save_norm = []

# 統一的：mouse callback function


def draw_roi(event, x, y, flags, param):
    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # 左鍵點擊，選擇點

        pts.append((x, y))
        print("pts:", pts)

        xx = x/img2.shape[1]
        yy = y/img2.shape[0]
        print("img_width:", img2.shape[1])
        print("img_height:", img2.shape[0])

        save_norm.append([xx, yy])
        print("ROI_points:", save_norm)

    if event == cv2.EVENT_RBUTTONDOWN:  # 右鍵點擊，取消最近一次選擇的點
        pts.pop()

    if event == cv2.EVENT_MBUTTONDOWN:  # 中鍵繪製輪廓
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
        # 畫多邊形
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # 用於求 ROI
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # 用於 顯示在桌面的圖像

        show_image = cv2.addWeighted(
            src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

        cv2.imshow("mask", mask2)
        cv2.imshow("show_img", show_image)

        ROI = cv2.bitwise_and(mask2, img)
        cv2.imshow("ROI", ROI)
        cv2.waitKey(0)

    if len(pts) > 0:
        # 將pts中的最後一點畫出來
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:
        # 畫線
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y 爲鼠標點擊地方的座標
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1],
                     color=(255, 0, 0), thickness=2)

    cv2.imshow('image', img2)


ROI = np.array([[0.257, 0.228], [0.060, 0.471], [0.112, 0.726], [0.187, 0.853], [
    0.407, 0.865], [0.457, 0.693], [0.443, 0.427]], np.float32)

# 創建圖像與窗口並將窗口與回調函數綁定
img = cv2.imread("./background.jpg")
# img = imutils.resize(img, width=500)
# Use for draw ROI polylines
width = img.shape[1]
height = img.shape[0]
img_size = np.array([[width, 0], [0, height]])
ROI_points = ROI.dot(img_size)
ROI_points = ROI_points.reshape((-1, 1, 2))

# Draw ROI area
cv2.polylines(img, pts=np.int32([ROI_points]), isClosed=True,
              color=(0, 0, 255), thickness=3)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_roi)
print("[INFO] 單擊左鍵：選擇點，單擊右鍵：刪除上一次選擇的點，單擊中鍵：確定ROI區域")
print("[INFO] 按‘S’確定選擇區域並保存")
print("[INFO] 按 ESC 退出")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("s"):
        saved_data = {
            "ROI": pts
        }

        # open file in write mode
        with open(r'./ROI_ponits.txt', 'w') as fp:

            fp.write("\n".join(str(item) for item in pts))

        # joblib.dump(value=saved_data, filename="config.txt")
        print("[INFO] ROI座標已保存到本地.")
        break
cv2.destroyAllWindows()
