import sys
import os
import socket
import shutil
import argparse

import numpy as np
import random
import cv2
from PIL import Image, ImageDraw, ImageFont

import moviepy.editor as mpy

from libDonkeyCar.datastore import Tub

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


def clamp(n, minn, maxn):
    return min(max(n, minn), maxn)


def mark_image(image, angle, throttle, mode):
    # make a frame with markings to show on video:
    video_img = Image.fromarray(image)
    pdraw = ImageDraw.Draw(video_img)

    img_width = image.shape[1]
    img_height = image.shape[0]

    line_width = 5  # of the elements in the large image
    color_angle = (0, 255, 0)
    color_throttle = (255, 0, 0)

    # throttle = random.random()      # 0...1

    pdraw.text((15, 10), "{:.3f}".format(angle), color_angle, font=ImageFont.truetype("arial", 16))
    pdraw.text((img_width - 60, 10), "{:.3f}".format(throttle), color_throttle, font=ImageFont.truetype("arial", 16))

    angle = clamp(angle, -1, 1)  # should be within the range anyway
    margin_top = 30
    margin_bottom = 10
    l = img_height - (margin_top + margin_bottom)
    dx = l * np.sin(angle)
    x = img_width / 2 + dx

    pdraw.line((x, margin_top, img_width / 2, img_height - margin_bottom), fill=color_angle, width=line_width)

    throttle = clamp(throttle, -1, 1)  # should be within the range anyway

    l = throttle * (img_height - (margin_top + margin_bottom))
    x = img_width - 5
    y = img_height - margin_bottom - l

    pdraw.line((x, y, x, img_height - margin_bottom), fill=color_throttle, width=line_width)

    return np.asarray(video_img, dtype=np.float32)


# This is called from the VideoClip as it references a time.
def make_frame(t):
    image, angle, throttle, mode = get_recorded_frame(t)
    # print(image[:20])
    # print(image.shape)  # (120, 160, 3)
    # print('angle=', angle, 'throttle=', throttle, 'mode=', mode)

    image_marked = mark_image(image, angle, throttle, mode)
    # print(image_marked[:20])
    # print(image_marked.shape)  # (120, 160, 3)

    # show movie:
    video_img_arr = image_marked / 255
    video_img_arr = cv2.cvtColor(video_img_arr, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR order
    cv2.imshow('Frame', video_img_arr)

    # Press Q on keyboard to  exit
    if cv2.waitKey(100) & 0xFF == ord('q'):
        return None  # AttributeError: 'NoneType' object has no attribute 'dtype', exits 1

    return image_marked  # image  # returns None or a 8-bit RGB array (120, 160, 3) required by mpy.VideoClip()


tub = Tub(path=path, inputs=inputs, types=types)
num_rec = tub.get_num_records()
iRec = 0

print('making movie:', out, 'from', num_rec, 'images')
clip = mpy.VideoClip(make_frame, duration=(num_rec // DRIVE_LOOP_HZ) - 1)
clip.write_videofile(out, fps=DRIVE_LOOP_HZ)

print('done')
