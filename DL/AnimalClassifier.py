import glob
import itertools
import os
import random
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Flatten, MaxPool2D)
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.simplefilter(action = 'ignore', category=FutureWarning)

# Cat vs Dog classifier. We use the kaggle datasets, preprocess it, organise it and then make batches of data to use it further.

# The accuracy after compiling is 100% but validation fails at 53%
#Epoch 10/10
#100/100 - 57s - loss: 0.0010 - accuracy: 1.0000 - val_loss: 2.5475 - val_accuracy: 0.5300

# O/P of this script is a 2*2 list. Interpret it as rows and column with titles "cats", "dogs".
# The prediction is [[28 22],[21 29]] which is not great. The model is over fitting.

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# This script resize all the images for keras.

train_path = "../Datasets/Training"
valid_path = "../Datasets/Valid"
test_path = "../Datasets/Testing"

# Preprocessing the images.
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

# Creating Neural network
model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2,2), strides=2),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2,2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# One thing to notice is that we are not specifying y=train_labels as while using ImageGenerator it stores labels in itself.
model.fit(x=train_batches, steps_per_epoch=len(train_batches), validation_data=valid_batches, validation_steps=len(valid_batches), epochs=10, verbose=2)

prediction = model.predict(x=test_batches, verbose=0)

# print(test_batches.class_indices) ==> {'cat': 0, 'dog': 1}
# test_batches.classes are labels (0's and 1's)
cm = confusion_matrix(y_true = test_batches.classes, y_pred = np.argmax(prediction, axis=-1))
# import seaborn as sns
# sns.heatmap(cm, annot = True)
plot_confusion_matrix(cm=cm, classes=['cat', 'dog'], title='Confusion matrix')


