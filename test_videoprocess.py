import unittest
import sys
import os
import numpy as np
import cv2

sys.path.append("..")
import pipeline
import videoprocess as uut

VIDEO_CLIP = "video_input_output/project_video.mp4"
TEST_OUT_DIR = "video_input_output/videoprocess"


class TestVideo(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(TEST_OUT_DIR):
            os.makedirs(TEST_OUT_DIR)

    # def test_ident(self):
    #     uut.process(VIDEO_CLIP, TEST_OUT_DIR + "/project_video.mp4", cb, subC=(3, 6))

    def test_pipe_project(self):
        p = pipeline.Pipeline()
        uut.process(VIDEO_CLIP, TEST_OUT_DIR + "/L_project_video.mp4", p.process)  # ,subC=(0,15))

    # def test_pipe_challenge(self):
    #     p = pipeline.Pipeline()
    #     uut.process(VIDEO_CLIP, TEST_OUT_DIR + "/L_challenge_video.mp4", p.process)  # ,subC=(0,15))
    #
    # def test_pipe_harder_challenge(self):
    #     p = pipeline.Pipeline()
    #     uut.process(VIDEO_CLIP, TEST_OUT_DIR + "/L_harder_challenge_video.mp4", p.process)  # ,subC=(0,15))


# def cb(img):
#     return img


if __name__ == '__main__':
    unittest.main()