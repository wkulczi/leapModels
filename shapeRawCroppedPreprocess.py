import keras.activations
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Reshape, Input, InputLayer
from keras.layers.convolutional import Conv2D, MaxPooling2D
import tensorflow as tf
import os
import random
import math
import shutil

data_dir = "../train/photoPadded/maxHeightNoResize/leapCropped"


def createTestDataset(seed=None):
    filePaths = []
    for dirname, dirs, files in os.walk(data_dir):
        for filename in files:
            filePaths.append(dirname[-1] + "/" + filename)

    random.seed(seed)
    randomSelect = random.sample(filePaths, math.floor(len(filePaths) * 0.2))

    for selectedFile in randomSelect:
        shutil.move(data_dir + "/" + selectedFile, "test/" + selectedFile)

    testFileNames = []

    for dirname, dirs, files in os.walk("test/"):
        for filename in files:
            testFileNames.append(dirname[-1] + "/" + filename)

    return testFileNames


def revertData(testFileNames):
    for testFilePath in testFileNames:
        try:
            shutil.move("test/" + testFilePath, data_dir + "/" + testFilePath)
        except Exception:
            pass


# todo zbuduj prepare data funkcje, która wydzieli osobny folder ze zdjęciami do zestawu testowego
# todo dodaj callbacki(?)


def to_grey_rescaled(image):
    # img = img_to_array(image)
    # img = tf.image.rgb_to_grayscale(img)

    img = image / 255
    return img


def path_to_image(path):
    img = tf.keras.utils.load_img(path, grayscale=True)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255
    return img


image_size = (171, 171)
batch_size = 32

testFileNames = createTestDataset()

testDatagen = ImageDataGenerator(preprocessing_function=to_grey_rescaled)

test_generator = testDatagen.flow_from_directory(
    'test',
    target_size=image_size,
    batch_size=batch_size,
    class_mode="sparse",
    color_mode='grayscale',
)

# https://www.geeksforgeeks.org/how-to-normalize-center-and-standardize-image-pixels-in-keras/
datagen = ImageDataGenerator(validation_split=0.2,
                             preprocessing_function=to_grey_rescaled)  # img/255 or use rescale=1.0/255.0

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="sparse",
    color_mode='grayscale',
    # https://stackoverflow.com/questions/59439128/what-does-class-mode-parameter-in-keras-image-gen-flow-from-directory-signify
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="sparse",
    color_mode='grayscale',
    # https://stackoverflow.com/questions/59439128/what-does-class-mode-parameter-in-keras-image-gen-flow-from-directory-signify
    subset='validation'
)

# czas matmę

# (N-F)/stride+1

model = Sequential()
model.add(
    Conv2D(filters=16, kernel_size=(3, 3), input_shape=(171, 171, 1), activation=keras.activations.relu,
           strides=(1, 1),
           kernel_initializer=keras.initializers.random_uniform))  # (171 + 2)-3/1+1 = 171
model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))  # 171-3/3 + 1 =  57
# model.add(
#     Conv2D(filters=32, kernel_size=(3, 3), input_shape=(171, 171, 1), padding="same", activation=keras.activations.relu,
#            strides=(1, 1),
#            kernel_initializer=keras.initializers.random_uniform))  # (57 + 2)-3/1+1 = 57
model.add(Flatten())  # 57*57 = 3249
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(.2))
model.add(Dense(11))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=3, validation_data=validation_generator)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('\nTest accuracy:', test_acc)

revertData(testFileNames)

# Test accuracy: 0.9917929172515869