import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

from libTestImageGen.testImageGen import TestImageGen
from libPlotter.testResultsPlotter import TestResultsPlotter

print(tf.__version__)

import platform
print(platform.python_version())

run_tag = "RunFromCheckpt"

# size_th = 28  # 64 # 28 # of the scaled down "thumbnail" that is passed to tensorflow model, e.x. 28
# class_names = ['left', 'left-corr', 'straight', 'right-corr', 'right', 'a', 'b', 'c', 'd', 'e']
# num_classes = len(class_names)


# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#
# train_labels = train_labels[:1000]
# test_labels = test_labels[:1000]
#
# train_images = train_images[:1000].reshape(-1, 28, 28, 1) / 255.0 #.reshape(-1, 28 * 28, 1) / 255.0
# test_images = test_images[:1000].reshape(-1, 28, 28, 1) / 255.0 #.reshape(-1, 28 * 28, 1) / 255.0


print('...creating model...')

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                        input_shape=(TestImageGen.size_th, TestImageGen.size_th, 1),
                        name='Conv2D_first'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Pooler_first'),
    keras.layers.Dropout(0.1, noise_shape=None, seed=None, name='Dropper_one'),
    keras.layers.Conv2D(64, (5, 5), activation='relu', name='Conv2D_second'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), name='Pooler_second'),
    keras.layers.Dropout(0.1, noise_shape=None, seed=None, name='Dropper_two'),
    keras.layers.Flatten(name='Flattener'),
    keras.layers.Dense(1000, activation='relu', name='Dense_Relu'),
    keras.layers.Dense(TestImageGen.num_classes, activation='softmax', name='Dense_Softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

print('FYI: checkpoints in:', checkpoint_dir)

model.load_weights(checkpoint_path)

# In[20]:

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

