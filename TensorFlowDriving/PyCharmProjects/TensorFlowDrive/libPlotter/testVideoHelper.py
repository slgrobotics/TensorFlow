import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class TestVideoHelper(object) :

    @staticmethod
    def clamp(n, minn, maxn):
        return min(max(n, minn), maxn)


    @staticmethod
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

        # draw a line in the middle, tilted to show wheels angle:
        angle = TestVideoHelper.clamp(angle, -1, 1)  # should be within the range anyway
        margin_top = 30
        margin_bottom = 10
        l = img_height - (margin_top + margin_bottom)
        dx = l * np.sin(angle)
        x = img_width / 2 + dx

        pdraw.line((x, margin_top, img_width / 2, img_height - margin_bottom), fill=color_angle, width=line_width)

        # draw a line on the right, to show throttle value:
        throttle = TestVideoHelper.clamp(throttle, -1, 1)  # should be within the range anyway

        l = throttle * (img_height - (margin_top + margin_bottom))
        x = img_width - 5
        y = img_height - margin_bottom - l

        pdraw.line((x, y, x, img_height - margin_bottom), fill=color_throttle, width=line_width)

        return np.asarray(video_img, dtype=np.float32)


    @staticmethod
    def showVideoFrame(image, waitms=25):
        # show one movie frame
        video_img_arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR order
        cv2.imshow('Frame', video_img_arr)

        # Press Q on keyboard to  exit - pass that to the caller:
        if cv2.waitKey(waitms) & 0xFF == ord('q'):
            return False  # AttributeError: 'NoneType' object has no attribute 'dtype', exits 1

        return True
