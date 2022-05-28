import os
import numpy as np
import cv2 as cv
import timeit
import matplotlib.pyplot as plt
from scipy import stats

algorithm = "BRISK"
path = 'C:\\Users\\taras.luchka\\Downloads\\magistr_dataset_with_train-val_tags_640_480\\images\\train\\'
arr = os.listdir(path)
f = open("{}_all_time_compute.txt".format(algorithm), "w")
number_of_kp = []
time_to_detect = []
time_to_compute = []
for a in arr:
    img1 = cv.imread('{}{}'.format(path,a), cv.IMREAD_GRAYSCALE)
    orb1 = cv.BRISK_create()
    start_detect = timeit.default_timer()
    #kp1, des1 = orb1.detectAndCompute(img1, None)
    kp1 = orb1.detect(img1,None)
    stop_detect = timeit.default_timer()

    start_compute = timeit.default_timer()
    kp1, des1 = orb1.compute(img1, kp1)
    stop_compute = timeit.default_timer()
    f.write("{} time to detect: {}; time to compute: {}; number of kp {}\n".format(a, round(stop_detect-start_detect, 5), round(stop_compute-start_compute, 5), len(kp1)))
    print('Time: ', stop_compute - start_detect) 
    number_of_kp.append(len(kp1))
    time_to_compute.append(round(stop_compute - start_compute, 5))
    time_to_detect.append(round(stop_detect - start_detect, 5)) 
    
combined_time = np.add(time_to_detect, time_to_compute)
print("corrcoef: {}\n".format(np.corrcoef(combined_time, number_of_kp)))
f.write("corrcoef: {}\n".format(np.corrcoef(combined_time, number_of_kp)))
f.write("avg time to detect : {}\n".format(np.mean(time_to_detect)))
f.write("avg time to compute : {}\n".format(np.mean(time_to_compute)))
f.write("avg time to detect and compute : {}\n".format(np.mean(time_to_compute + time_to_detect)))
f.write("avg number of kp : {}\n".format(np.mean(number_of_kp)))

f.write("mode time to detect : {}\n".format(stats.mode(time_to_detect)))
f.write("mode time to compute : {}\n".format(stats.mode(time_to_compute)))
f.write("mode time to detect and compute : {}\n".format(stats.mode(time_to_detect + time_to_compute)))
f.write("mode number of kp : {}\n".format(stats.mode(number_of_kp)))

f.write("median time to detect : {}\n".format(np.median(time_to_detect)))
f.write("median time to compute : {}\n".format(np.median(time_to_compute)))
f.write("median time to detect and compute : {}\n".format(np.median(time_to_detect + time_to_compute)))
f.write("median number of kp : {}\n".format(np.median(number_of_kp)))
f.close()


#---------------------combined plots-------------------------
fig, ax1 = plt.subplots()
plt.title("{}".format(algorithm))

color = 'tab:red'
ax1.set_xlabel('number')
ax1.set_ylabel('number of key points', color=color)
ax1.plot(range(0, len(number_of_kp)), number_of_kp, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('time to compute', color=color)  # we already handled the x-label with ax1
ax2.plot(range(0, len(time_to_compute)), time_to_compute, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

#plt.show()
plt.savefig("{}_all_time_to_compute_and_number_together.png".format(algorithm), dpi=600)
plt.close(fig)

#2
fig, ax1 = plt.subplots()
plt.title("{}".format(algorithm))

color = 'tab:red'
ax1.set_xlabel('number')
ax1.set_ylabel('number of key points', color=color)
ax1.plot(range(0, len(number_of_kp)), number_of_kp, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('time to detect', color=color)  # we already handled the x-label with ax1
ax2.plot(range(0, len(time_to_compute)), time_to_detect, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

#plt.show()
plt.savefig("{}_all_time_to_detect_and_number_together.png".format(algorithm), dpi=600)
plt.close(fig)

#3
fig, ax1 = plt.subplots()
plt.title("{}".format(algorithm))

color = 'tab:red'
ax1.set_xlabel('number')
ax1.set_ylabel('number of key points', color=color)
ax1.plot(range(0, len(number_of_kp)), number_of_kp, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('time to detect and compute', color=color)  # we already handled the x-label with ax1
ax2.plot(range(0, len(time_to_compute)), combined_time, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

#plt.show()
plt.savefig("{}_all_time_to_detect_and_compute_and_number_together.png".format(algorithm), dpi=600)
plt.close(fig)





#---------------------------individual plots--------------------------------
plt.plot(number_of_kp)
plt.xlabel('number')
plt.ylabel('number of key points')
plt.title("{}".format(algorithm))
plt.savefig("{}_all_number_of_kp.png".format(algorithm), dpi=600)
plt.close()

plt.plot(time_to_compute)
plt.xlabel('number')
plt.ylabel('s')
plt.title("{}".format(algorithm))
plt.savefig("{}_all_time_to_compute.png".format(algorithm), dpi=600)
plt.close()

plt.plot(time_to_detect)
plt.xlabel('number')
plt.ylabel('s')
plt.title("{}".format(algorithm))
plt.savefig("{}_all_time_to_detect.png".format(algorithm), dpi=600)
plt.close()

plt.plot(combined_time)
plt.xlabel('number')
plt.ylabel('s')
plt.title("{}".format(algorithm))
plt.savefig("{}_all_time_to_detect_and_compute.png".format(algorithm), dpi=600)
plt.close()