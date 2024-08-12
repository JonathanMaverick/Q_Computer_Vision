import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def image_processing(): 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_file_path = os.path.join(script_dir, 'Santa.jpg')
    img = cv2.imread(image_file_path)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray_image)
    cv2.waitKey(0)

    _, bin_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    _, inv_bin_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    _, trunc_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_TRUNC)
    _, tozero_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_TOZERO)
    _, inv_tozero_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_TOZERO_INV)
    _, otsu_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_OTSU)

    result_image = [gray_image, bin_thresh, inv_bin_thresh, trunc_thresh, tozero_thresh, inv_tozero_thresh, otsu_thresh]
    result_desc = ['Grayscale', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'OTSU']
    plt.figure(1, figsize=(8, 8))
    for i, (curr_image, curr_desc) in enumerate(zip(result_image, result_desc)):
        plt.subplot(3, 3, (i+1))
        plt.imshow(curr_image, 'gray')
        plt.title(curr_desc)
        plt.xticks([])
        plt.yticks([])
    plt.show()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    mean_blur = cv2.blur(img, (11, 11))
    gaussian_blur = cv2.GaussianBlur(img, (11, 11), 5.0) 
    median_blur = cv2.medianBlur(img, 11) 
    bilateral_blur = cv2.bilateralFilter(img, 5, 150, 150) 

    result_image = [img, mean_blur, gaussian_blur, median_blur, bilateral_blur]
    result_desc = ['Original', 'Mean', 'Gaussian', 'Median', 'Bilateral']
    plt.figure(2, figsize=(8, 8))
    for i, (curr_image, curr_desc) in enumerate(zip(result_image, result_desc)):
        plt.subplot(3, 3, (i+1))
        plt.imshow(curr_image)
        plt.title(curr_desc)
        plt.xticks([])
        plt.yticks([])
    plt.show()