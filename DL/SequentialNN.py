import keras
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# We are testing out a drug on people between the age 18-80. It's been observed that mostly people below 50, 
# don't experience side effect but people above 50 do. 

train_samples, train_labels, test_samples, test_labels, scaled_train_samples, scaled_test_samples = [], [], [], [], [], []


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

def initTrainingData():
    global train_samples, train_labels, test_samples, test_labels, scaled_train_samples, scaled_test_samples
    for i in range(1000):
        # Young people's data
        train_samples.append(randint(18,50))
        train_labels.append(0)
        test_samples.append(randint(18,50))
        test_labels.append(0)
        # Old  people's data
        train_samples.append(randint(50,80))
        train_labels.append(1)
        test_samples.append(randint(50,80))
        test_labels.append(1)

    # Convert into numpy array
    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)
    test_samples = np.array(test_samples)
    test_labels = np.array(test_labels)

    # Shuffle data
    train_samples, train_labels = shuffle(train_samples, train_labels)
    test_samples, test_labels = shuffle(test_samples, test_labels)

    # Scale the age between 0-1
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
    scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

    print("Training data created successfully.")

initTrainingData()

# Training
model = Sequential([
    Dense(16, activation="relu", input_shape=(1,)),
    Dense(32, activation="relu"),
    Dense(2, activation="sigmoid")
])
model.compile(Adam(lr=0.0001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(scaled_train_samples, train_labels, batch_size=10, epochs=20, validation_split=0.20, shuffle=True, verbose=0)

# Testing
predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)

# Visualization
rounded_predictions = np.argmax(predictions, axis= -1) # To select the most probable predicitons i.e. the one with high value. It's 1's and 0's i.e. affected or not affected
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
plot_confusion_matrix(cm=cm, classes=['Not affected', 'Affected'],title="Confusion matrix")