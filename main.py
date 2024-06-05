import os
import pickle

import cv2
import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

img_bg = cv2.imread('Resources/background.png')

folder_mode_path = 'Resources/Modes'
mode_path_list = os.listdir(folder_mode_path)
img_mode_list = []
for path in mode_path_list:
    img_mode_list.append(cv2.imread(os.path.join(folder_mode_path, path)))

# Load the encoding file
with open("EncodeFile.p", "rb") as file:
    encode_list_known_with_ids = pickle.load(file)
encode_list_known, student_ids = encode_list_known_with_ids
print(student_ids)
while True:
    success, img = cap.read()
    img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    face_cur_frame = face_recognition.face_locations(img_s)
    encode_cur_frame = face_recognition.face_encodings(img_s, face_cur_frame)

    img_bg[162:162+480, 55:55+640] = img
    img_bg[44:44 + 633, 808:808 + 414] = img_mode_list[2]

    for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        print("matches", matches)
        print("face_dis", face_dis)
    cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", img_bg)
    cv2.waitKey(1)

