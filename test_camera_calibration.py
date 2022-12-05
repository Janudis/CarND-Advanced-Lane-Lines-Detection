import sys
import os
import unittest
import cv2

sys.path.append("..")
import image_util
import calibration as uut

CAMERA_CAL_DIR = 'data_cal\\camera_cal/calibration*.jpg'
TEST_IMAGES_DIR = 'data_test\\test_images/*.jpg'
TEST_OUT_DIR = "output_images\\camera_calibration_out"


class CameraClibrationTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(TEST_OUT_DIR):
            os.makedirs(TEST_OUT_DIR)

    def tearDown(self):
        return

    def test_01_calibrateCameraFromDir(self):
        mtx, dist = uut.calibration()
        print(mtx, dist)

    def test_02_undistortCalibrationImages(self):
        imgs = image_util.loadImagesRGB(CAMERA_CAL_DIR)
        mtx, dist = uut.calibration()
        for i, img in enumerate(imgs):
            dimg = cv2.undistort(img, mtx, dist)
            image_util.saveBeforeAfterImages(img, "Original", dimg, "Undistorted",
                                             TEST_OUT_DIR + "/calibration_" + str(i) + "_undistorted.png")

    def test_03_undistortTestImages(self):
        imgs = image_util.loadImagesRGB(TEST_IMAGES_DIR)
        mtx, dist = uut.calibration()
        for i, img in enumerate(imgs):
            dimg = cv2.undistort(img, mtx, dist)
            image_util.saveBeforeAfterImages(img, "Original", dimg, "Undistorted",
                                             TEST_OUT_DIR + "/test_" + str(i) + "_undistorted.png")


if __name__ == '__main__':
    unittest.main()