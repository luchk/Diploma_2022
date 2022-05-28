import os
import numpy as np
import cv2 as cv
import timeit

algorithm = "ORB"
path = 'C:\\Users\\taras.luchka\\Downloads\\Diploma\\image samples\\'
arr = os.listdir(path)
f = open("{}_single_log.txt".format(algorithm), "w")
for a in arr:
    img1 = cv.imread('{}{}'.format(path,a), cv.IMREAD_GRAYSCALE)
    orb1 = cv.ORB_create(10000)
    start_detect = timeit.default_timer()
    #kp1, des1 = orb1.detectAndCompute(img1, None)
    kp1 = orb1.detect(img1,None)
    stop_detect = timeit.default_timer()

    start_compute = timeit.default_timer()
    kp1, des1 = orb1.compute(img1, kp1)
    stop_compute = timeit.default_timer()
    f.write("{} time to detect: {}; time to compute: {}; number of kp {}\n".format(a, round(stop_detect-start_detect, 5), round(stop_compute-start_compute, 5), len(kp1)))
    print('Time: ', stop_compute - start_detect) 
    draw = cv.drawKeypoints(img1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imwrite("{}_key_points_single_{}.jpeg".format(algorithm, a),draw)
f.close()


 
