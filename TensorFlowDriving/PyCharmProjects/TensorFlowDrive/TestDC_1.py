import sys
import os
import socket
import shutil
import argparse

import moviepy.editor as mpy

from libDonkeyCar.datastore import Tub

path = 'C:\\Projects\\Robotics\\DonkeyCar\\DonkeySimWindows\\log'
DRIVE_LOOP_HZ = 10
out = 'movie.mp4'


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

    image = rec['cam/image_array']  # a 8-bit RGB array, shape: (120, 160, 3)
    angle = rec['user/angle']
    throttle = rec['user/throttle']
    mode = rec['user/mode']

    return image, angle, throttle, mode  # returns a tuple


inputs = ["cam/image_array", "user/angle", "user/throttle", "user/mode"]  # ['user/speed', 'cam/image']
types = ["image_array", "float", "float", "str"]  # ['float', 'image']


# This is called from the VideoClip as it references a time.
def make_frame(t):
    image, angle, throttle, mode = get_recorded_frame(t)
    # print(image[:20])
    # print(image.shape)  # (120, 160, 3)
    # print('angle=', angle, 'throttle=', throttle, 'mode=', mode)
    return image  # returns None or a 8-bit RGB array (120, 160, 3) required by mpy.VideoClip()


tub = Tub(path=path, inputs=inputs, types=types)
num_rec = tub.get_num_records()
iRec = 0

print('making movie:', out, 'from', num_rec, 'images')
clip = mpy.VideoClip(make_frame, duration=(num_rec // DRIVE_LOOP_HZ) - 1)
clip.write_videofile(out, fps=DRIVE_LOOP_HZ)

print('done')
