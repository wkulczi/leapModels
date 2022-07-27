import keras.activations
# datapath = "../train/photoPadded/maxHeightNoResize/leapCropped/"
# fileNames = [file for file in
#              os.listdir(datapath) if
#              os.path.isfile(os.path.join(datapath, file))]
#
# classes = list(sorted(set([x[0] for x in fileNames])))
#
# data = []
# y = []
#
# for fileName in fileNames:
#     img = load_img(f"{datapath}/{fileName}", color_mode='rgb')
#     img = img_to_array(img)
#     img = tf.image.rgb_to_grayscale(img)
#     data.append(img)
#     y.append(classes.index(fileName[0]))
#
# print(y)
#

#todo zbuduj prepare data funkcje, która wydzieli osobny folder ze zdjęciami do zestawu testowego
#todo dodaj callbacki(?)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Reshape, Input, InputLayer
from keras.layers.convolutional import Conv2D, MaxPooling2D


def to_grey_rescaled(image):
    # img = img_to_array(image)
    # img = tf.image.rgb_to_grayscale(img)

    img = image / 255
    return img


image_size = (171, 171)
batch_size = 32

data_dir = "../train/photoPadded/maxHeightNoResize/leapCropped"
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
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(171,171,1), padding="same", activation=keras.activations.relu, strides=(1,1),
                 kernel_initializer=keras.initializers.random_uniform))                               #(171 + 2)-3/1+1 = 171
model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3)))                                               #171-3/3 + 1 =  57
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(171,171,1), padding="same", activation=keras.activations.relu, strides=(1,1),
                 kernel_initializer=keras.initializers.random_uniform))                               #(57 + 2)-3/1+1 = 57
model.add(Flatten())                                                                                  #57*57 = 3249
model.add(Dense(64, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(11))


model.summary()

model.compile(optimizer='adam', loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=10, validation_data=validation_generator)


# 550/550 [==============================] - 146s 264ms/step - loss: 0.2123 - accuracy: 0.9316 - val_loss: 0.0437 - val_accuracy: 0.9861
# Epoch 2/10
# 550/550 [==============================] - 146s 266ms/step - loss: 0.0461 - accuracy: 0.9871 - val_loss: 0.0271 - val_accuracy: 0.9884
# Epoch 3/10
# 550/550 [==============================] - 148s 269ms/step - loss: 0.0238 - accuracy: 0.9931 - val_loss: 0.0170 - val_accuracy: 0.9945
# Epoch 4/10
# 550/550 [==============================] - 146s 266ms/step - loss: 0.0148 - accuracy: 0.9960 - val_loss: 0.0615 - val_accuracy: 0.9857
# Epoch 5/10
# 550/550 [==============================] - 141s 256ms/step - loss: 0.0086 - accuracy: 0.9976 - val_loss: 0.0633 - val_accuracy: 0.9852
# Epoch 6/10
# 550/550 [==============================] - 139s 252ms/step - loss: 0.0114 - accuracy: 0.9967 - val_loss: 0.0191 - val_accuracy: 0.9943
# Epoch 7/10
# 550/550 [==============================] - 143s 259ms/step - loss: 0.0084 - accuracy: 0.9971 - val_loss: 0.0488 - val_accuracy: 0.9891
# Epoch 8/10
# 550/550 [==============================] - 136s 247ms/step - loss: 0.0093 - accuracy: 0.9970 - val_loss: 0.0329 - val_accuracy: 0.9916
# Epoch 9/10
# 550/550 [==============================] - 139s 253ms/step - loss: 0.0046 - accuracy: 0.9986 - val_loss: 0.0262 - val_accuracy: 0.9923
# Epoch 10/10
# 550/550 [==============================] - 119s 217ms/step - loss: 0.0045 - accuracy: 0.9985 - val_loss: 0.0354 - val_accuracy: 0.9930
#
# Process finished with exit code 0

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size,
#     epochs=nb_epochs)
