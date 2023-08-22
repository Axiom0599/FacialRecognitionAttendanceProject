import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime


video_capture = cv2.VideoCapture(0)

vinisha_image = face_recognition.load_image_file("photos/vinisha.jpg")
vinisha_encoding = face_recognition.face_encodings(vinisha_image)[0]


nelli_image = face_recognition.load_image_file("photos/nelli.jpg")
nelli_encoding = face_recognition.face_encodings(nelli_image)[0]

shraddha_image = face_recognition.load_image_file("photos/shraddha.jpg")
shraddha_encoding = face_recognition.face_encodings(shraddha_image)[0]

maryrose_image = face_recognition.load_image_file("photos/Maryrose.png")
maryrose_encoding = face_recognition.face_encodings(maryrose_image)[0]

known_face_encoding = [
    vinisha_encoding,
    nelli_encoding,
    shraddha_encoding,
    maryrose_encoding
]

known_face_names = [
    "Vinisha",
    "Nelli",
    "Shraddha",
    "Mary Rose"
]

students = known_face_names.copy()

face_location = []
face_encodings = []
face_names = []
s= True


dateTime = datetime.now()
current_date = dateTime.strftime("%d-%m-%y")

f = open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)


while True:
    _,frame= video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])


    if s:
        face_location = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_location)
        face_names = []


        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)


            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print("recorded")
                    print(students)
                    current_time = dateTime.strftime("%D-%M-%Y")
                    lnwriter.writerow([name,current_time])

            else:
                print("student doesn't exist")





video_capture.release()
cv2.destroyAllWindows()
f.close()
