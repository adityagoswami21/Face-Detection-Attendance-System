import os
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

img_bg = cv2.imread('Resources/background.png')

folder_mode_path = 'Resources/Modes'
mode_path_list = os.listdir(folder_mode_path)
img_mode_list = []
for path in mode_path_list:
    img_mode_list.append(cv2.imread(os.path.join(folder_mode_path, path)))
while True:
    success, img = cap.read()

    img_bg[162:162+480, 55:55+640] = img
    cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", img_bg)
    cv2.waitKey(1)

