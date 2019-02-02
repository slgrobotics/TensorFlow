import sys
import os
import socket
import shutil
import argparse

import numpy as np
import random
import cv2

import moviepy.editor as mpy
from PIL import Image, ImageDraw, ImageFont

from libDonkeyCar.datastore import Tub

from libPlotter.testVideoHelper import TestVideoHelper

path = 'C:\\Projects\\Robotics\\DonkeyCar\\DonkeySimWindows\\log'  # Sim or actual drive log location
DRIVE_LOOP_HZ = 10
out = 'movie.mp4'
outnpz = 'train_images.npz'
size_th = 28 # 64 # 32 # of the scaled down "thumbnail" that is passed to tensorflow model, e.x. 28

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

    listnpz.append((image, angle, throttle))

    return image_marked  # image  # returns None or a 8-bit RGB array (120, 160, 3) required by mpy.VideoClip()


def make_npz(listnpz, outnpz):
    print('len(listnpz)=', len(listnpz))

    images = []
    labels = []
    angles = []
    throttles = []

    class_names = ['left', 'left-corr', 'straight', 'right-corr', 'right']

    for item in listnpz:

        image = item[0]
        angle = item[1]
        throttle = item[2]

        angles.append(angle)
        throttles.append(throttle)

        # print(image.shape, angle, throttle)  # (120, 160, 3) 0.0035258643329143524 0.30000001192092896

        video_img = Image.fromarray(image)
        img_w = video_img.width
        img_h = video_img.height
        video_img = video_img.crop((0,img_h/2,img_w,img_h))

        thumb_img = video_img.resize((size_th, size_th)).convert(mode='L')  # to grayscale
        thumb_arr = np.asarray(thumb_img, dtype=np.float32)

        # print(thumb_arr.shape)  # (28, 28)
        thumb_arr = thumb_arr.reshape((size_th, size_th, 1))
        # print(thumb_arr.shape)  # (28, 28, 1)

        images.append(thumb_arr)

        label = 2 # class_names[2]  # 'straight'
        if(angle > 0.5):
            label = 4 # class_names[4]
        elif(angle < -0.5):
            label = 0 # class_names[0]
        elif(angle > 0.1):
            label = 3 # class_names[3]
        elif(angle < -0.1):
            label = 1 # class_names[1]

        labels.append(label)

    # we need something like this:   train_images.shape: (8000, 28, 28, 1)
    # test_images, test_labels =

    images_arr = np.asarray(images)
    labels_arr = np.asarray(labels)
    angles_arr = np.asarray(angles)
    throttles_arr = np.asarray(throttles)

    print(images_arr.shape)
    print(labels_arr.shape)
    print(labels_arr[:10])

    print('...saving to file:', outnpz)
    np.savez(outnpz, images_arr, labels_arr, angles_arr, throttles_arr)

    return

tub = Tub(path=path, inputs=inputs, types=types)
num_rec = tub.get_num_records()
iRec = 0
listnpz = []

print('making movie:', out, 'from', num_rec, 'images')
try:
    clip = mpy.VideoClip(make_frame, duration=(num_rec // DRIVE_LOOP_HZ) - 1)
    clip.write_videofile(out, fps=DRIVE_LOOP_HZ)
except:
    print('caught exception')
finally:
    make_npz(listnpz, outnpz)

print('done')
