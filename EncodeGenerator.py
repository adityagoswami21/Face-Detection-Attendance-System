import cv2
import face_recognition
import os
import pickle

folder_path = 'Images'
path_list = os.listdir(folder_path)
print(path_list)
img_list = []
student_ids = []
for path in path_list:
    img_list.append(cv2.imread(os.path.join(folder_path, path)))
    student_ids.append(os.path.splitext(path)[0])
print(len(img_list))
