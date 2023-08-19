import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

s = True
video_capture = cv2.VideoCapture(0)

percy_image = face_recognition.load_image_file("photos/percy.jpeg")
percy_encoding = face_recognition.face_encodings(percy_image)[0]


annabeth_image = face_recognition.load_image_file("photos/annabeth.png")
annabeth_encoding = face_recognition.face_encodings(annabeth_image)[0]


grover_image = face_recognition.load_image_file("photos/grover.jpeg")
grover_encoding = face_recognition.face_encodings(grover_image)[0]


known_face_encoding = [
    percy_encoding,
    annabeth_encoding,
    grover_encoding
]

known_face_names = [
    "Percyy Jackson",
    "Annabeth Chase",
    "Grover Underwood"
]
# ... (imports and known_face_encodings and names)

students = known_face_names.copy()

# ... (initialize CSV file)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = datetime.strftime("%D-%M-%Y")
                    lnwriter.writerow([name, current_time])
                else:
                    print("student doesn't exist")

        # Display the frame with recognized faces
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the processed frame
        cv2.imshow('Video', frame)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
f.close()
