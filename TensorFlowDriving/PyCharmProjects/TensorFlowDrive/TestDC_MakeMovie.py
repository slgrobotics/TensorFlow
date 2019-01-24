import sys
import os
import socket
import shutil
import argparse

import moviepy.editor as mpy

from libDonkeyCar.datastore import Tub

def make_frame(t):
    """
    Callback to return an image from from our tub records.
    This is called from the VideoClip as it references a time.
    We don't use t to reference the frame, but instead increment
    a frame counter. This assumes sequential access.
    """
    global iRec

    iRec = iRec + 1

    if iRec >= num_rec - 1:
        return None

    rec = tub.get_record(iRec)
    image = rec['cam/image_array']

    return image  # returns a 8-bit RGB array

path = 'C:\\Projects\\Robotics\\DonkeyCar\\DonkeySimWindows\\log'
inputs = [ "cam/image_array","user/angle","user/throttle","user/mode" ] # ['user/speed', 'cam/image']
types = [ "image_array","float","float","str" ] # ['float', 'image']

tub = Tub(path=path, inputs=inputs, types=types)
num_rec = tub.get_num_records()
iRec = 0
DRIVE_LOOP_HZ = 10
out = 'movie.mp4'

print('making movie', out, 'from', num_rec, 'images')
clip = mpy.VideoClip(make_frame, duration=(num_rec//DRIVE_LOOP_HZ) - 1)
clip.write_videofile(out ,fps=DRIVE_LOOP_HZ)

print('done')

