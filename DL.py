import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
import numpy as np


# Weight and height of man and women.
train_samples = np.array([[200, 65], [180, 50], [150, 60], [210, 70], [200, 80], [190, 70]])
train_labels = np.array([1, 0, 0, 1, 1, 0])  # 1 means female, 0 means male.
test_samples = np.array([[189, 64], [170, 51], [130, 59], [200, 70], [210, 70], [191, 75]])

model = Sequential([
    Dense(16, activation="relu", input_shape=(2,)),
    Dense(32, activation="relu"),
    Dense(2, activation="sigmoid")
])
model.compile(Adam(lr=0.001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(train_samples, train_labels, batch_size=3, epochs=20, validation_split=0.20, shuffle=True, verbose=1)
predictions = model.predict(test_samples, verbose=0)

# Code doesnt work.
for i in predictions:
    print(predictions)
