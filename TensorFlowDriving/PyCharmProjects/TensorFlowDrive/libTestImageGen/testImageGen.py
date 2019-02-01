from PIL import Image, ImageDraw
import numpy as np

import random

size = 256  # of the "large" image emulating camera frame

lane_width = 80  # width of the "road" in pixels on the large image

noise_circles = 5  # 5
noise_dots = 500  # 500

color_bg = (255, 255, 255, 255)
color_fg = (0, 0, 0, 255)
color_outline = (255, 255, 255, 255)
color_fill = (255, 255, 255, 127)

img_size = (size, size)
poly_size = (size, size)
poly_offset = (0, 0)  # location in base image
polygon = [(0, 0), (0, size), (size, size), (size, 0)]
line_width = 5  # of the elements in the large image

class_names_dictionary = {
    0: 'left', 1: 'left-corr', 2: 'straight', 3: 'right-corr', 4: 'right'
}


class TestImageGen(object):

    class_names = ['left', 'left-corr', 'straight', 'right-corr', 'right']

    num_classes = len(class_names)

    # makes one large RGBA image, simulating camera frame
    @staticmethod
    def make_image(dir):
        mode = 'RGBA'
        back_img = Image.new(mode, img_size, color_bg)
        poly_img = Image.new(mode, poly_size)
        pdraw = ImageDraw.Draw(poly_img)
        pdraw.polygon(polygon, fill=color_fill, outline=color_outline)

        px = random.randint(1, 100)
        py = random.randint(1, 100)
        pxx = random.randint(-10, 10)
        pyy = random.randint(1, 100)
        pxmid = random.randint(120, 132)  # around the center point
        pymid = random.randint(120, 132)
        dx = random.randint(20, 100)  # for left and right correction
        dy = 0

        # draw the two-line lane to be recognized:
        for i in range(-1, 1):
            l_width = lane_width / 2 * i
            y_corr = l_width * 3 / 4
            if dir == 0:  # left
                pdraw.line((pxmid + l_width, pymid - y_corr, px + l_width, py - y_corr), fill=color_fg,
                           width=line_width)
                pdraw.line((pxmid + l_width, pymid - y_corr, pxmid + l_width, size - 5 - pyy - y_corr), fill=color_fg,
                           width=line_width)
            elif dir == 1:  # left correction
                pdraw.line((pxmid - dx + l_width, pymid + dy, pxmid + pxx - dx + l_width, py + dy), fill=color_fg,
                           width=line_width)
                pdraw.line((pxmid - dx + l_width, pymid + dy, pxmid - dx + l_width, size - 5 - pyy + dy), fill=color_fg,
                           width=line_width)
            elif dir == 2:  # straight
                pdraw.line((pxmid + l_width, pymid, pxmid + pxx + l_width, py), fill=color_fg, width=line_width)
                pdraw.line((pxmid + l_width, pymid, pxmid + l_width, size - 5 - pyy), fill=color_fg, width=line_width)
            elif dir == 3:  # right correction
                pdraw.line((pxmid + dx + l_width, pymid + dy, pxmid + pxx + dx + l_width, py + dy), fill=color_fg,
                           width=line_width)
                pdraw.line((pxmid + dx + l_width, pymid + dy, pxmid + dx + l_width, size - 5 - pyy + dy), fill=color_fg,
                           width=line_width)
            else:  # 4 # right
                pdraw.line((pxmid + l_width, pymid + y_corr, size - 5 - px + l_width, py + y_corr), fill=color_fg,
                           width=line_width)
                pdraw.line((pxmid + l_width, pymid + y_corr, pxmid + l_width, size - 5 - pyy + y_corr), fill=color_fg,
                           width=line_width)

        if noise_circles > 0:
            # draw some noise, circles randomly:
            for i in range(0, random.randint(0, noise_circles)):
                r = random.randint(10, 30)
                x = random.randint(0, size) + r
                y = random.randint(0, size) + r
                pdraw.ellipse((x - r, y - r, x + r, y + r), width=line_width, fill=(0, 0, 0, 0), outline=color_fg)

        if noise_dots > 0:
            # draw some noise, dots randomly:
            for i in range(0, random.randint(0, noise_dots)):
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                pdraw.line((x, y, x - line_width, y - line_width), fill=color_fg, width=line_width)

        back_img.paste(poly_img, poly_offset, mask=poly_img)
        return back_img

    # In[5]:

    # converts image to a (size_th x size_th) numpy array, suitable for TensorFlow input
    @staticmethod
    def to_image_arr(img, size_th):
        img.thumbnail((size_th, size_th))
        img = img.convert("RGB")
        # img.show()
        img_arr = np.asarray(img, dtype=np.float32) / 255
        img_arr = img_arr[:, :, :1]
        # img_arr = img_arr.reshape((size_th, size_th))
        # img_arr.shape

        # img_arr = np.zeros(shape=(size_th, size_th), dtype=np.float32)
        return img_arr

    # In[6]:

    # creates train or test set - images (as size_th x size_th numpy arrays) and random labels
    # see https://www.tensorflow.org/tutorials/keras/basic_classification
    @staticmethod
    def all_images(num_images, size_th = 32):
        # size_th = 32 # 64 # 28 # of the scaled down "thumbnail" that is passed to tensorflow model, e.x. 28
        np.random.seed(420976534)
        labels = np.random.choice([0, 1, 2, 3, 4], size=num_images)
        images = np.empty(shape=(num_images, size_th, size_th, 1), dtype=np.float32)
        for i in range(0, num_images):
            img = TestImageGen.make_image(labels[i])
            img_arr = TestImageGen.to_image_arr(img, size_th)
            images[i] = img_arr
        return images, labels  # images.shape (num_images, size_th, size_th, 1)
