import sys
import os
import socket
import shutil
import argparse

import numpy as np
import cv2
from PIL import Image, ImageDraw

import moviepy.editor as mpy

from libDonkeyCar.datastore import Tub

path = 'C:\\Projects\\Robotics\\DonkeyCar\\DonkeySimWindows\\log'   # Sim or actual drive log location
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
    angle = rec[inputs[1]]
    throttle = rec[inputs[2]]
    mode = rec[inputs[3]]

    return image, angle, throttle, mode  # returns a tuple


# converts image to a (size_th x size_th) numpy array, suitable for TensorFlow input
def to_image_arr(img):
    img = img.convert("RGB")
    # img.show()
    img_arr = np.asarray(img, dtype=np.float32) / 255
    img_arr = img_arr[:, :, :1]
    # img_arr = img_arr.reshape((size_th, size_th))
    # img_arr.shape

    # img_arr = np.zeros(shape=(size_th, size_th), dtype=np.float32)
    return img_arr


def mark_image(image, angle, throttle, mode):
    # make a frame with markings to show on video:
    shape = image.shape

    video_img = Image.fromarray(np.uint8(image))
    pdraw = ImageDraw.Draw(video_img)

    line_width = 10  # of the elements in the large image
    color_fg = (0, 127, 0)

    pdraw.line((10, 10, 80, 80), fill=color_fg, width=line_width)

    video_img_arr = to_image_arr(video_img)  # to a numpy array

    return np.stack([video_img_arr,video_img_arr,video_img_arr], axis=2).reshape((shape[0], shape[1], 3))


# This is called from the VideoClip as it references a time.
def make_frame(t):
    image, angle, throttle, mode = get_recorded_frame(t)
    # print(image[:20])
    # print(image.shape)  # (120, 160, 3)
    # print('angle=', angle, 'throttle=', throttle, 'mode=', mode)

    video_img_arr = mark_image(image, angle, throttle, mode)
    # print(video_img_arr[:20])
    # print(video_img_arr.shape)  # (120, 160, 3)

    # show movie:
    cv2.imshow('Frame', video_img_arr)

    # Press Q on keyboard to  exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        return None     # AttributeError: 'NoneType' object has no attribute 'dtype', exits 1

    return video_img_arr * 255  # image  # returns None or a 8-bit RGB array (120, 160, 3) required by mpy.VideoClip()


tub = Tub(path=path, inputs=inputs, types=types)
num_rec = tub.get_num_records()
iRec = 0

print('making movie:', out, 'from', num_rec, 'images')
clip = mpy.VideoClip(make_frame, duration=(num_rec // DRIVE_LOOP_HZ) - 1)
clip.write_videofile(out, fps=DRIVE_LOOP_HZ)

print('done')
