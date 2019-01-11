import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

from libTestImageGen.testImageGen import TestImageGen
from libPlotter.testResultsPlotter import TestResultsPlotter

print(tf.__version__)

import platform
print(platform.python_version())

run_tag = "RunFromCheckptHDF5"

run_tag_while_trained = "conv11"

print('...restoring previously saved model...')

saved_model_filename = run_tag_while_trained + '_model.hdf5'

model = keras.models.load_model(saved_model_filename)

# https://www.tensorflow.org/tutorials/keras/save_and_restore_models
# if model was using optimizers from tf.train, you need to compile the model manually
#    you will loose the state of the optimizer.
# if keras optimizer was used (e.g. optimizer="adam") - compile step is not needed

# model.compile(optimizer=tf.train.AdamOptimizer(),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

print('...making test_images...')

test_images, test_labels = TestImageGen.all_images(1000)

print('...evaluating...')

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

predictions[0]

# tf.summary.histogram("predictions", predictions)

np.argmax(predictions[0])

out_filename = run_tag + '_out.png'

print('...plotting to file: ', out_filename)

# see folder C:\Users\sergei\PycharmProjects\TensorFlowDrive

TestResultsPlotter.plotResults(out_filename, predictions, test_labels, test_images)

print('OK: finished')

