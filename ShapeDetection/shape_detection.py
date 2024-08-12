import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def shape_detection():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'image.png')

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    harris_result = cv2.cornerHarris(gray, 2, 5, 0.04)

    cv2.imshow('Harris', harris_result)
    cv2.waitKey(0)

    without_subpix = img.copy()
    without_subpix[harris_result > 0.01*harris_result.max()] = [0, 0, 255]

    _, thresh = cv2.threshold(harris_result, 0.01*harris_result.max(), 255, 0)
    thresh = np.uint8(thresh)

    _, _, _, corner_centroids = cv2.connectedComponentsWithStats(thresh)
    corner_centroids = np.float32(corner_centroids)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.001)

    enhanced_corners = cv2.cornerSubPix(gray, corner_centroids, (2, 2), (-1, -1), criteria)

    with_subpix = img.copy()
    corner_centroids = np.int16(corner_centroids)
    for centroid in corner_centroids:
        centroid_y = centroid[1]
        centroid_x = centroid[0]

        with_subpix[centroid_y, centroid_x] = [0, 0, 255]
        
    enhanced_corners = np.int16(enhanced_corners)
    for corner in enhanced_corners:
        corner_y = corner[1]
        corner_x = corner[0]
        
        with_subpix[corner_y, corner_x] = [0, 255, 0]

    plt.figure(1, figsize=(7, 7))

    without_subpix = cv2.cvtColor(without_subpix, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.imshow(without_subpix)
    plt.title('Without subpix')
    plt.xticks([])
    plt.yticks([])

    with_subpix = cv2.cvtColor(with_subpix, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 2)
    plt.imshow(with_subpix)
    plt.title('With subpix')
    plt.xticks([])
    plt.yticks([])

    plt.show()

    fast = cv2.FastFeatureDetector_create()
    fast_keypoints = fast.detect(img, None)
    fast_result = img.copy()
    cv2.drawKeypoints(img, fast_keypoints, fast_result, color=[0, 0, 255])
    fast_result = cv2.cvtColor(fast_result, cv2.COLOR_BGR2RGB)

    orb = cv2.ORB_create()
    orb_keypoints = orb.detect(img, None)
    orb_result = img.copy()
    cv2.drawKeypoints(img, orb_keypoints, orb_result, color=[255, 0, 0])
    orb_result = cv2.cvtColor(orb_result, cv2.COLOR_BGR2RGB)

    combined_result = img.copy()
    cv2.drawKeypoints(img, fast_keypoints, combined_result, color=[0, 0, 255]) 
    cv2.drawKeypoints(combined_result, orb_keypoints, combined_result, color=[255, 0, 0]) 

    combined_result = cv2.cvtColor(combined_result, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 2, 1)
    plt.imshow(fast_result)
    plt.title('FAST Keypoints (Red)')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(orb_result)
    plt.title('ORB Keypoints (Blue)')
    plt.xticks([])
    plt.yticks([])

    plt.show()