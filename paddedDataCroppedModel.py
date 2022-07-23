import numpy as np
from tensorflow.python import keras
import tensorflow as tf
from tensorflow.python.keras import layers
import os
from keras_preprocessing.image import load_img, img_to_array

datapath = "../train/photoPadded/maxHeightNoResize/"
fileNames = [file for file in
             os.listdir(datapath) if
             os.path.isfile(os.path.join(datapath, file))]

classes = list(sorted(set([x[0] for x in fileNames[0]])))

data = []
y = []

for fileName in fileNames:
    img = load_img(f"{datapath}/{fileName}", color_mode='rgb')
    img = img_to_array(img)
    img = tf.image.rgb_to_grayscale(img)
    data.append(img)
    y.append(classes.index(fileName[0]))




