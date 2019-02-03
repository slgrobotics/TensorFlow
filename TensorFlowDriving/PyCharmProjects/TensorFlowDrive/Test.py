import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

from time import time
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import sklearn.model_selection as sk        # https://scikit-learn.org/stable/
import os

from libTestImageGen.testImageGen import TestImageGen
from libPlotter.testResultsPlotter import TestResultsPlotter
from libPlotter.testVideoHelper import TestVideoHelper

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

run_tag = "convDC12"
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
    #train_labels = npzfile['arr_1']
    train_angles = npzfile['arr_2']
    # train_throttles = npzfile['arr_3']

print('...making val_images...')

# val_images, val_labels = TestImageGen.all_images(1000, size_th)
train_images, val_images, train_angles, val_angles = \
    sk.train_test_split(train_images, train_angles,test_size=0.2, random_state = 42)

print('...making test_images...')

# test_images, test_labels = TestImageGen.all_images(1000, size_th)
train_images, test_images, train_angles, test_angles = \
    sk.train_test_split(train_images, train_angles,test_size=0.2, random_state = 42)

print('train_images.shape:', train_images.shape)

print('train_angles:', train_angles[:100])

num_rows = 50
num_cols = 8
line_width = 1  # of the elements in the large image
color_angle = 0
#color_throttle = 255

num_images = num_rows * num_cols
fig = plt.figure(figsize=(2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    angle = train_angles[i]

    image = train_images[i].reshape((size_th, size_th))
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

    pdraw.line((x, margin_top, img_width / 2, img_height - margin_bottom),
               fill=color_angle, width=line_width)

    plt.imshow(np.asarray(video_img, dtype=np.float32), cmap=plt.cm.binary)
    plt.xlabel("{:.3f}".format(angle))

out_filename = run_tag + '_befr_out.png'

fig.savefig(out_filename, bbox_inches='tight')

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
    #keras.layers.Dense(TestImageGen.num_classes, activation='softmax', name='Dense_Softmax'),
    keras.layers.Dense(units=1, activation='linear', name='angle_out') #,
    # keras.layers.Dense(units=1, activation='linear', name='throttle_out')
])

# use keras optimizer (e.g. optimizer="adam") - so that compile step is not
#     needed when model is loaded from HDF5 file
# https://www.tensorflow.org/tutorials/keras/save_and_restore_models

model.compile(optimizer="adam", # tf.train.AdamOptimizer(),
              loss={'angle_out': 'mean_squared_error'},
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

model.fit(train_images, train_angles,
          epochs=EPOCHS,
          validation_data=(val_images, val_angles),
          verbose=1, callbacks=[tensorboard_callback, checkpoint_callback])

model.save(filepath=run_tag + '_model.hdf5', include_optimizer=True, overwrite=True)

test_loss, test_acc = model.evaluate(test_images, test_angles)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

predictions[0]

# tf.summary.histogram("predictions", predictions)

np.argmax(predictions[0])

out_filename = run_tag + '_aftr_out.png'

print('...plotting to file: ', out_filename)

# see folder C:\Users\sergei\PycharmProjects\TensorFlowDrive

TestResultsPlotter.plotResults(out_filename, predictions, test_angles, test_images)

print('OK: finished')
