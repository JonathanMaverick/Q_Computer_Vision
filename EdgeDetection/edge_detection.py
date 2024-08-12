import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def edge_detection() : 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_file_path = os.path.join(script_dir, 'image.jpg')
    img = cv2.imread(image_file_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    laplacian = cv2.Laplacian(gray, cv2.CV_64F) 
    laplacian_uint8 = np.uint8(np.absolute(laplacian))

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 5)

    canny = cv2.Canny(gray, 100, 200)

    result_image = [gray, laplacian, laplacian_uint8, sobel_x, sobel_y, canny]
    result_desc = ['Original', 'Laplacian', 'uint8', 'Sobel_X', 'Sobel_Y', 'Canny']
    plt.figure(1, figsize=(8, 8))
    for i, (curr_image, curr_desc) in enumerate(zip(result_image, result_desc)):
        plt.subplot(2, 3, (i+1))
        plt.imshow(curr_image, 'gray')
        plt.title(curr_desc)
        plt.xticks([])
        plt.yticks([])
    plt.show()

