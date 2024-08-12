import cv2
import os

def face_detection():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    haarcascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            detected = frame[y:y + h, x: x + w]
            detected = cv2.GaussianBlur(detected, (23, 23), 30)
            frame[y:y + h, x: x + w] = detected

            cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27: 
            break
        
    cap.release()
    cv2.destroyAllWindows()