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
print(student_ids)


def find_encodings(images_list):
    encode_list = []
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list


print("Encode Started ...")
encode_list_known = find_encodings(img_list)
encode_list_known_with_ids = [encode_list_known, student_ids]
print("Encoding Complete")

with open("EncodeFile.p", "wb") as file:
    pickle.dump(encode_list_known_with_ids, file)


