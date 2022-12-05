import unittest
import os
import sys
import numpy as np
import cv2

sys.path.append("..")
import pipeline
import image_util

IMG_DIR = 'data_test\\test_images/*.jpg'
TEST_OUT_DIR = "output_images\\pipeline"


class TestPipeline(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(TEST_OUT_DIR):
            os.makedirs(TEST_OUT_DIR)

    def test_process(self):
        p = pipeline.Pipeline()
        imgs = image_util.loadImagesRGB(IMG_DIR)
        for i, img in enumerate(imgs):
            processed_img = p.process(img)
            image_util.saveBeforeAfterImages(img, "Original", processed_img, "Thresholded",
                                             TEST_OUT_DIR + "/thresholded" + str(i) + ".jpg")


if __name__ == '__main__':
    unittest.main()