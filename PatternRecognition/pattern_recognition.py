import cv2
import os
import math
import numpy as np

def preprocess_face(face_img):
    face_img = cv2.equalizeHist(face_img) 
    face_img = cv2.resize(face_img, (200, 200))  
    return face_img

def pattern_recognition() :
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, './images/train')
    person_names = os.listdir(train_path)
    
    face_cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    
    face_list = []
    class_list = []
    for index, person_name in enumerate(person_names):
        full_name_path = train_path + '/' + person_name

        for image_path in os.listdir(full_name_path):
            full_image_path = full_name_path + '/' + image_path
            img_gray = cv2.imread(full_image_path, 0)

            detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=2, minNeighbors=5)
            
            if(len(detected_faces) < 1):
                continue

            for face_rect in detected_faces:
                x, y, w, h = face_rect
                face_img = img_gray[y:y+w, x:x+h]
                face_img = preprocess_face(face_img)
                face_list.append(face_img)
                class_list.append(index)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_list, np.array(class_list))

    test_path = os.path.join(script_dir, './images/test')
    for image_path in os.listdir(test_path):
        full_image_path = test_path + '/' + image_path
        img_bgr = cv2.imread(full_image_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=3)

        if(len(detected_faces) < 1):
            continue

        for face_rect in detected_faces:
            x, y, w, h = face_rect
            face_img = img_gray[y:y+w, x:x+h]

            res, confidence = face_recognizer.predict(face_img)
            confidence = math.floor(confidence * 100) / 100

            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 1)
            text = person_names[res] + ' ' + str(confidence) + '%'
            cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
            cv2.imshow('res', img_bgr)
            cv2.waitKey(0)