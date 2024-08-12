import cv2
import os

def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    cascade_file_path = os.path.join(script_dir, cascade_file)
    image_file_path = os.path.join(script_dir, filename)

    cascade = cv2.CascadeClassifier(cascade_file_path)
    image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(0)

def detect_anime(): 
    detect('image.png')

