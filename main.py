from AnimeFaceDetection import anime_detection
from EdgeDetection import edge_detection
from ImageProcessing import image_processing
from PatternRecognition import pattern_recognition
from ShapeDetection import shape_detection
from FaceDetection import face_detection
        

def main():
    print("1. Anime Detection")
    print("2. Edge Detection")
    print("3. Image Processing")
    print("4. Pattern Recognition")
    print("5. Shape Detection")
    print("6. Face Recognition")
    print("7. Exit")
    
    while True:
        try:
            choice = int(input(">> "))
        except ValueError:
            print("Input tidak valid. Masukkan angka 1, 2, atau 3.")
            continue
        
        if choice == 1:
            anime_detection.detect_anime()
        elif choice == 2:
            edge_detection.edge_detection()
        elif choice == 3:
            image_processing.image_processing()
        elif choice == 4:
            pattern_recognition.pattern_recognition()
        elif choice == 5 : 
            shape_detection.shape_detection()
        elif choice == 6:
            face_detection.face_detection()
        elif choice == 7:
            print("Bye bye")
            break
        else:
            print("Input invalid")

if __name__ == "__main__":
    main()