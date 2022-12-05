import pickle
import cv2
import calibration
import os
import matplotlib.pyplot as plt

# the camera_cal images are expected relatively to THIS python file

class Undistorter:

    def __init__(self):
        dist_pickle = pickle.load(open("data_cal\\camera_cal/wide_dist_pickle.p", "rb"))
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist)

# img = cv2.imread('test_images/straight_lines1.jpg')
# a = Undistorter()
# dsted = a.undistort(img)
# plt.imshow(dsted)
# plt.show()
