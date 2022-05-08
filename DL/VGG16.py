import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import backend as k
from sklearn.preprocessing import LabelBinarizer

import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import itertools


warnings.simplefilter(action='ignore', category=FutureWarning)

(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

# We are only going to classify dog and cat.
cifar_classes = {
    #     0: "airplane",
    #     1: "automobile",
    #     2: "bird",
    3: "cat",
    #     4: "deer",
    5: "dog",
    #     6: "frog",
    #     7: "horse",
    #     8: "ship",
    #     9: "truck"
}

# Creates a 1D array for iterable obj. e.g [3,5]
class_list = np.fromiter(cifar_classes.keys(), float)
# Reshapes our [3,5] => [[3], [5]]
class_list = np.reshape(class_list, (class_list.size, 1))

# Extracting only limited data from the dataset. i.e. only required classes
# np.isin...flatten() Returns [True, False, ...] based on if class_list element matches train_labels element.
train_data = train_data[np.isin(train_labels, class_list).flatten()]
train_labels = train_labels[np.isin(train_labels, class_list).flatten()]

test_data = test_data[np.isin(test_labels, class_list).flatten()]
test_labels = test_labels[np.isin(test_labels, class_list).flatten()]

# Scaling pixel values to [0-1]
train_data = train_data.astype('float')/255
test_data = test_data.astype('float')/255

# Turning the [[2], [1],...] labels into [[0,1], [1,0]] One-hot encoding vectors
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)
model = Sequential([
    # 1st block
    Conv2D(filters=32, kernel_size=(3, 3),
           padding='same', input_shape=(32, 32, 3)),
    Activation('relu'),
    BatchNormalization(axis=-1),

    Conv2D(filters=32, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    BatchNormalization(axis=-1),

    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 2nd block
    Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    BatchNormalization(axis=-1),

    Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    BatchNormalization(axis=-1),

    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Fully connected layer
    Flatten(),
    Dense(units=512),
    Activation('relu'),
    BatchNormalization(),
    Dropout(0.5),

    # O/P layer
    Dense(units=2),
    Activation('softmax')
])

model.summary()

model.compile(optimizer=SGD(learning_rate=0.01,
              decay=0.01/40, momentum=0.9, nesterov=True))
model.fit(x=train_data, y=train_labels, validation_data=(
    test_data, test_labels),  batch_size=64, epochs=40, verbose=1)
