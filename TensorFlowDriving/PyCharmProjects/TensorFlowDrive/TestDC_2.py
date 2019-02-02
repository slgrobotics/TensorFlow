import sys
import os
import socket
import shutil
import argparse

import numpy as np
import random
import cv2

import moviepy.editor as mpy

from libDonkeyCar.datastore import Tub

from libPlotter.testVideoHelper import TestVideoHelper

path = 'C:\\Projects\\Robotics\\DonkeyCar\\DonkeySimWindows\\log'  # Sim or actual drive log location
DRIVE_LOOP_HZ = 10
out = 'movie.mp4'

inputs = ["cam/image_array", "user/angle", "user/throttle", "user/mode"]  # ['user/speed', 'cam/image']
types = ["image_array", "float", "float", "str"]  # ['float', 'image']


def get_recorded_frame(t):
    """
    Callback to return an image from from our tub records.
    This is indirectly called from the VideoClip as it references a time.
    We don't use t to reference the frame, but instead increment
    a frame counter. This assumes sequential access.
    """
    global iRec

    iRec = iRec + 1

    if iRec >= num_rec - 1:
        return None

    rec = tub.get_record(iRec)

    image = rec[inputs[0]]  # a 8-bit RGB array, shape: (120, 160, 3)
    angle = rec[inputs[1]]  # range -1.0...1.0
    throttle = rec[inputs[2]]  # range 0...1.0
    mode = rec[inputs[3]]  # 'user'

    return image, angle, throttle, mode  # returns a tuple


# This is called from the VideoClip as it references a time.
def make_frame(t):
    image, angle, throttle, mode = get_recorded_frame(t)
    # print(image[:20])
    # print(image.shape)  # (120, 160, 3)
    # print('angle=', angle, 'throttle=', throttle, 'mode=', mode)

    image_marked = TestVideoHelper.mark_image(iRec, image, angle, throttle, mode)
    # print(image_marked[:20])
    # print(image_marked.shape)  # (120, 160, 3)

    # show movie frame:
    if(TestVideoHelper.showVideoFrame(image_marked / 255, waitms=10) == False):
        # Pressed Q on keyboard to exit
        return None  # AttributeError: 'NoneType' object has no attribute 'dtype', exits 1

    return image_marked  # image  # returns None or a 8-bit RGB array (120, 160, 3) required by mpy.VideoClip()


tub = Tub(path=path, inputs=inputs, types=types)
num_rec = tub.get_num_records()
iRec = 0

print('making movie:', out, 'from', num_rec, 'images')
clip = mpy.VideoClip(make_frame, duration=(num_rec // DRIVE_LOOP_HZ) - 1)
clip.write_videofile(out, fps=DRIVE_LOOP_HZ)

print('done')
