import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def shape_detection():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    object_path = os.path.join(script_dir, './images/object.jpg')
    scene_path = os.path.join(script_dir, './images/scene.jpg')

    img_object = cv2.imread(object_path)
    img_scene = cv2.imread(scene_path)

    surf = cv2.SIFT_create()

    kp_object, des_object = surf.detectAndCompute(img_object, None)
    kp_scene, des_scene = surf.detectAndCompute(img_scene, None)

    des_object = des_object.astype('f')
    des_scene = des_scene.astype('f')

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_object, des_scene, 2)

    matchesMask = []
    for i in range(0, len(matches)):
        matchesMask.append([0, 0])

    total_match = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            total_match += 1

    img_res = cv2.drawMatchesKnn(
        img_object, kp_object, img_scene,
        kp_scene, matches, None,
        matchColor = [0, 255, 0], 
        singlePointColor = [255, 0, 0], 
        matchesMask = matchesMask
    )
    plt.imshow(img_res)
    plt.show()