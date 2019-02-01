import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

from time import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk        # https://scikit-learn.org/stable/
import os

from libTestImageGen.testImageGen import TestImageGen
from libPlotter.testResultsPlotter import TestResultsPlotter

print(tf.__version__)

import platform
print(platform.python_version())

#matplotlib.use('TkAgg')

# xxx=matplotlib.is_interactive()
# print(xxx)
#
# matplotlib.get_backend()
# matplotlib.interactive(True)
#
# xxx=matplotlib.is_interactive()
# print(xxx)

# In[3]:

run_tag = "conv11"
size_th = 28 # 64 # 28 # of the scaled down "thumbnail" that is passed to tensorflow model, e.x. 28

EPOCHS = 3

filename = 'train_images.npz'

if(os.path.isfile(filename) != True):
    print('...no file - making train_images...')
    train_images, train_labels = TestImageGen.all_images(10000, size_th)
    print('...saving to file: ', filename)
    np.savez(filename, train_images, train_labels)
else:
    print('...loading train_images from file:', filename)
    npzfile = np.load(filename)
    train_images = npzfile['arr_0']
    train_labels = npzfile['arr_1']

print('...making val_images...')

# val_images, val_labels = TestImageGen.all_images(1000, size_th)
train_images, val_images, train_labels, val_labels = \
    sk.train_test_split(train_images, train_labels,test_size=0.2, random_state = 42)

print('...making test_images...')

# test_images, test_labels = TestImageGen.all_images(1000, size_th)
train_images, test_images, train_labels, test_labels = \
    sk.train_test_split(train_images, train_labels,test_size=0.2, random_state = 42)

print('train_images.shape:', train_images.shape)

for i in range(0, 5):
    plt.figure()
    plt.imshow(train_images[i].reshape((size_th, size_th)))  # train_images[0])
    plt.colorbar()
    plt.grid(False)

train_labels[:100]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape((size_th, size_th)), cmap=plt.cm.binary)
    plt.xlabel(TestImageGen.class_names[train_labels[i]])

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(size_th, size_th), name='Flattener'),
#     keras.layers.Dense(128, activation=tf.nn.relu, name='Dense_Relu'),
#     keras.layers.Dropout(0.1, noise_shape=None, seed=None, name='Dropper'),
#     keras.layers.Dense(10, activation=tf.nn.softmax, name='Dense_Softmax')
# ])

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                        input_shape=(size_th, size_th, 1),
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

# use keras optimizer (e.g. optimizer="adam") - so that compile step is not
#     needed when model is loaded from HDF5 file
# https://www.tensorflow.org/tutorials/keras/save_and_restore_models

model.compile(optimizer="adam", # tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# in the PyCharm environment terminal, run:
#     C:\Users\sergei\AppData\Local\conda\conda\envs\TensorFlow_Py3_6\Scripts\tensorboard --logdir logs/1
# point the browser here:
#     http://localhost:6006

ts = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
log_path = "logs/1/" + ts + "_-_" + run_tag  # + model.name

tensorboard_callback = TensorBoard(log_path, histogram_freq=1, write_graph=True, write_grads=True, write_images=True)

# docs:  https://keras.io/callbacks/#tensorboard
# logs in C:\Users\sergei\Documents\TensorFlowDriving\logs\1

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_images, train_labels,
          epochs=EPOCHS,
          validation_data=(val_images, val_labels),
          verbose=1, callbacks=[tensorboard_callback, checkpoint_callback])

model.save(filepath=run_tag + '_model.hdf5', include_optimizer=True, overwrite=True)

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
