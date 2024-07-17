import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def mark_attendance(name):
    with open('Attendance.csv', 'r+') as f:  # Open Attendance.csv for reading and writing
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt_string}')  # Write attendance record if not present

# Path to the images folder
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    cur_img = cv2.imread(f'{path}/{cl}')  # Read image
    images.append(cur_img)
    classNames.append(os.path.splitext(cl)[0])  # Extract image name

encode_list_known = find_encodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()

    # Resize image for faster processing
    img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    # Find faces and their encodings
    faces_cur_frame = face_recognition.face_locations(img_s)
    encodes_cur_frame = face_recognition.face_encodings(img_s, faces_cur_frame)

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):

        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        match_index = np.argmin(face_dis)
        if matches[match_index]:
            name = classNames[match_index].upper()
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Adjust for resizing

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            mark_attendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
