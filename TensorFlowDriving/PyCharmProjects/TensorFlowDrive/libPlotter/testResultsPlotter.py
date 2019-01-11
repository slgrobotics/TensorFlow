import numpy as np
import matplotlib.pyplot as plt

from libTestImageGen.testImageGen import TestImageGen

class TestResultsPlotter(object) :

    @staticmethod
    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img.reshape((TestImageGen.size_th, TestImageGen.size_th)), cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(TestImageGen.class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             TestImageGen.class_names[true_label]),
                   color=color)


    @staticmethod
    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(TestImageGen.num_classes), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')


    @staticmethod
    def plotResults(out_filename, predictions, test_labels, test_images):
        # Plot the first X test images, their predicted label, and the true label
        # Color correct predictions in blue, incorrect predictions in red
        num_rows = 50
        num_cols = 4
        num_images = num_rows * num_cols
        fig=plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            TestResultsPlotter.plot_image(i, predictions, test_labels, test_images)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            TestResultsPlotter.plot_value_array(i, predictions, test_labels)

        #plt.show()
        fig.savefig(out_filename,bbox_inches='tight')
