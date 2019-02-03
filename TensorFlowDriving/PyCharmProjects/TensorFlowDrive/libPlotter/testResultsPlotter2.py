import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from libPlotter.testVideoHelper import TestVideoHelper

#
# to plot results of a Donkey Car dataset, with angles and throttle as floats
#
class TestResultsPlotter2(object) :

    @staticmethod
    def mark_image(image, angle, color_angle=0):

        img_width = image.shape[1]
        img_height = image.shape[0]

        video_img = Image.fromarray(image)
        pdraw = ImageDraw.Draw(video_img)

        # pdraw.text((1, 0), "{:.3f}".format(angle), color_angle, font=ImageFont.truetype("arial", 9))

        # draw a line in the middle, tilted to show wheels angle:
        angle = TestVideoHelper.clamp(angle, -1, 1)  # should be within the range anyway
        margin_top = 3
        margin_bottom = 1
        l = img_height - (margin_top + margin_bottom)
        dx = l * np.sin(angle)
        x = img_width / 2 + dx

        line_width = 1

        pdraw.line((x, margin_top, img_width / 2, img_height - margin_bottom),
                   fill=color_angle, width=line_width)

        return np.asarray(video_img, dtype=np.float32)

    @staticmethod
    def plot_image(i, predictions_array, img, test_angles, test_throttles):
        predictions_array, true_angle, img = predictions_array[i], test_angles[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        size_th = img.shape[0]

        predicted_angle = np.argmax(predictions_array)

        video_img = TestResultsPlotter2.mark_image(img.reshape((size_th, size_th)), true_angle)
        video_img = TestResultsPlotter2.mark_image(video_img, predicted_angle, color_angle=255)
        plt.imshow(video_img, cmap=plt.cm.binary)

        if predicted_angle == true_angle:
            color = 'blue'
        else:
            color = 'red'

        max_pred = 100 * np.max(predictions_array)
        plt.xlabel("{:1.3f} ({:1.3f})".format(predicted_angle, true_angle), color=color)


    @staticmethod
    def plot_value_array(i, predictions_array, test_angles, test_throttles):
        predictions_array, true_label = predictions_array[i], test_angles[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(50), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        #thisplot[predicted_label].set_color('red')
        #thisplot[true_label].set_color('blue')

    @staticmethod
    def plotBefore(out_filename, train_images, train_angles, test_throttles):
        num_rows = 50
        num_cols = 8
        line_width = 1  # of the elements in the large image
        color_angle = 0
        # color_throttle = 255

        num_images = num_rows * num_cols
        fig = plt.figure(figsize=(2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)

            size_th = train_images[i].shape[0]
            angle = train_angles[i]
            video_img = TestResultsPlotter2.mark_image(train_images[i].reshape((size_th, size_th)), angle)

            plt.imshow(video_img, cmap=plt.cm.binary)
            plt.xlabel("{:.3f}".format(angle))

        fig.savefig(out_filename, bbox_inches='tight')

    @staticmethod
    def plotResults(out_filename, predictions, test_images, test_angles, test_throttles):
        # Plot the first X test images, their predicted label, and the true label
        # Color correct predictions in blue, incorrect predictions in red
        num_rows = 50
        num_cols = 4
        num_images = num_rows * num_cols
        fig=plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            TestResultsPlotter2.plot_image(i, predictions, test_images, test_angles, test_throttles)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            TestResultsPlotter2.plot_value_array(i, predictions, test_angles, test_throttles)

        #plt.show()
        fig.savefig(out_filename,bbox_inches='tight')
