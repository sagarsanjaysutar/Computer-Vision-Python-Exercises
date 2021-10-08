from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# Program to train and test our NN model on MNIST digit dataset.
# Final o/p will be a table which shoes digits 0-9 and how precisly our model has predicted.

print("[INFO] Loading MNIST dataset...")
digits = datasets.load_digits();
data = digits.data.astype('float') # [[1,2,3...64], [1,2,3...64],...1797 times...]
data = (data - data.min()) / (data.max() - data.min()) # Scaling the values between 0-1
print("[INFO] Samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

(train_data, test_data, train_label, test_label) = train_test_split(data, digits.target, test_size=0.25) # digits.target is numpy array of labels.

train_label = LabelBinarizer().fit_transform(train_label) # Converting the test label from 1d to 2d.
test_label = LabelBinarizer().fit_transform(test_label)

print("[INFO] Training network...")
nn = NeuralNetwork([train_data.shape[1], 32, 16, 10]) # 64-32-16-10 arch. 1st layer as 64 nodes, 2nd has 32... till the o/p layer which has 10 nodes as digits are 0-9 
print("[INFO] {}".format(nn))
nn.fit(train_data, train_label, epochs = 1000)

print("[INFO] Evaluating network...")
prediction = nn.predict(test_data)
prediction = prediction.argmax(axis=1)
print(classification_report(test_label.argmax(axis=1), prediction))

# print(type(train_data), type(test_data), type(train_label), type(test_label))
# print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)

