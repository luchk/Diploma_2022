import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import timeit

algorithm = "BIRSK"

path = 'C:\\Users\\taras.luchka\\Downloads\\Diploma\\image samples\\'
img1_name = "0AB74F96-54BF-4CEE-894E-339D53674E13.jpeg"
img2_name = "4FA72988-D339-4BB8-9EBA-9E424B24CF89.jpeg"
img3_name = "5D8F6183-6352-4ACD-B66A-2A1DA99A834C.jpeg"
img4_wrong_name = "0CCB6460-B44C-4BE8-80DE-3250AAFC6F6C.jpeg"

img_list = [img2_name, img3_name, img4_wrong_name]

f = open("{}_all_time_compute_match.txt".format(algorithm), "w")

for a in img_list:
    img1 = cv.imread('{}{}'.format(path, img1_name),cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread('{}{}'.format(path, a),cv.IMREAD_GRAYSCALE) # trainImage

    sift = cv.BRISK_create()

    # start_detectAndCompute = timeit.default_timer()
    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)
    # stop_detectAndCopute = timeit.default_timer()
    # print(stop_detectAndCopute - start_detectAndCompute)


    start_detect = timeit.default_timer()
    kp1 = sift.detect(img1, None)
    kp2 = sift.detect(img2, None)
    stop_detect = timeit.default_timer()

    start_compute = timeit.default_timer()
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)
    stop_compute = timeit.default_timer()

    time_to_detect_and_compute = (stop_detect - start_detect) + (stop_compute - start_compute)
    # BFMatcher with default params
    bf = cv.BFMatcher()

    start_knnMatch_and_find_good = timeit.default_timer()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    stop_knnMatch_and_find_good = timeit.default_timer()
    print(stop_knnMatch_and_find_good - start_knnMatch_and_find_good)
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite("{}_match_{}_to_{}.jpeg".format(algorithm, img1_name, a), img3)
    f.write("{} to {} detect time: {}; compute time: {}; detect and compute time: {}; match time: {}; number of good matches: {}; number of kp1: {}; number of kp 2: {}\n".format(img1_name, a, round(stop_detect - start_detect, 5), round(stop_compute - start_compute, 5), round(time_to_detect_and_compute, 5), round(stop_knnMatch_and_find_good - start_knnMatch_and_find_good, 5), len(good), len(kp1), len(kp2)))


f.close()