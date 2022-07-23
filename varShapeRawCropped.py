import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Reshape, Input, InputLayer
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras_preprocessing.image import load_img, img_to_array
import os
from sklearn.model_selection import train_test_split
import cv2

# this is just to unconfuse pycharm
try:
    from cv2 import cv2
except ImportError:
    pass


class DataGen(keras.utils.Sequence):

    def __init__(self, IDs, labels, data_directory, batch_size=32, n_classes=10, shuffle=True):
        self.IDs = IDs
        self.labels = labels
        self.dataDirectory = data_directory
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = []  # indexes of files that will be picked in batch
        self.max_dimension = None
        self.onEpochEnd()

    def onEpochEnd(self):
        self.indexes = np.arange(len(self.IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.IDs) / float(self.batch_size)))

    def __getitem__(self, index):  # generate batch

        print(index)
        rangeStart = index * self.batch_size
        rangeEnd = index + 1 * self.batch_size
        indexes = self.indexes[rangeStart:rangeEnd]

        pickedIDs = [self.IDs[i] for i in indexes]
        print(pickedIDs)
        X, y = self.__generate_data(pickedIDs)
        return X, y

    def __load_image(self, imageName):
        img = load_img(f"{self.dataDirectory}/{imageName}", color_mode='rgb')
        img = img_to_array(img)
        img = tf.image.rgb_to_grayscale(img)

        max_dim = max(img.shape)
        if self.max_dimension:
            if max_dim > self.max_dimension:
                new_dim = tuple(dim * self.max_dimension // max_dim for dim in img.shape[1::-1])
                img = cv2.resize(img, new_dim)

        img = img / 255

        return img

    @staticmethod
    def __pad_images(image, maxres):
        return np.pad(image, (*[((maxres[i] - image.shape[i]) // 2,
                                 ((maxres[i] - image.shape[i]) // 2) + ((maxres[i] - image.shape[i]) % 2)) for i in
                                range(2)],
                              (0, 0)), mode='constant', constant_values=0.)

    def __generate_data(self, pickedIDs):
        Xs = [self.__load_image(imageID) for imageID in pickedIDs]
        Ys = [self.labels[imageID] for imageID in pickedIDs]

        maxres = tuple(max([img.shape[i] for img in Xs]) for i in range(2))
        Xs = np.array([self.__pad_images(image, maxres) for image in Xs])

        return Xs, keras.utils.to_categorical(Ys, num_classes=self.n_classes)


# load data and classes
datapath = "../train/photoRaw/leapCropped"
fileNames = [file for file in
             os.listdir(datapath) if
             os.path.isfile(os.path.join(datapath, file))]

classes = sorted(list(set([x[0] for x in fileNames])))

fileClasses = {x: classes.index(x[0]) for x in fileNames}
# split data into train and test

X_train, X_test, _, _ = train_test_split(list(fileClasses.keys()),
                                                                 list(fileClasses.values()), test_size=0.2)

# define data generators
params = {
    'data_directory': datapath,
    'batch_size': 64,
    'n_classes': 11,
    'shuffle': True}

training_gen = DataGen(X_train, fileClasses, **params)
validation_gen = DataGen(X_test, fileClasses, **params)

# define model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='random_uniform'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='random_uniform'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(training_gen, epochs = 2, shuffle=False)

# train model
