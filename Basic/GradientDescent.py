import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Applying gradient descent on following data
# Feature vector: [3.4, 4.2],[3.12, 2.2].... Label: 1,0,0,1,..
# O/P of the script shows losses in each epochs, which tends to reduce every epoch. 
# as well as a report of how well our predictions were as compared to true values.

# Activation function which turns the input "ON"(activate) or "OFF"
def sigmoid_activation(input):
    return 1/(1+np.exp(-input)) # Sigmoid activation formula

def predict(feature_vector, weight_matrix):
    # X: Input data(Image) or feature vector
    # W: Weight matrix
    pred_label = sigmoid_activation(feature_vector.dot(weight_matrix))
    # 0 and 1 are labels
    pred_label[pred_label <= 0.5] = 0
    pred_label[pred_label > 0.5] = 1
    return pred_label

# Parsing Arguements sent from cmd
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100)
ap.add_argument("-a", "--alpha", type=float, default=0.01)
args = vars(ap.parse_args())

# Data generation
# X holds feature data i.e. [[-3.7, 4.5],..] and y holds labels for each items in X as 1 or 0. Both X and y are numpy array. 
(feature_vector, label) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1) #(numberOfItems, column, ?, standardDeviation, randomState)
label = label.reshape((label.shape[0], 1)) # y.shape returns size of y in tuple, so to get first element we use [0]. y.reshape(1000,1)
feature_vector = np.c_[feature_vector, np.ones(feature_vector.shape[0])] # To combine both weight & bias we insert a bias column of 1 in our feature vector. Concatenation of 2 list i.e X and [1,1,..1]

#Split data
(train_FV, test_FV, train_label, test_label) = train_test_split(feature_vector, label, test_size=0.5, random_state=42) # FV - feature vector

print("[INFO] training...")
weight_matrix = np.random.randn(feature_vector.shape[1], 1) # Weight matrix
losses = []

for epoch in np.arange(0, args["epochs"]):
    pred_label = sigmoid_activation(train_FV.dot(weight_matrix))
    error = pred_label - train_label # Error is difference between our predictions and "true" values.
    loss = np.sum(error**2)
    losses.append(loss)
    gradient = train_FV.T.dot(error) # Since we now have error, we compute gradients of to update our weight. T - transpose
    weight_matrix += -args["alpha"] * gradient # Weight-update-rule: Our weight matrix will keep getting updated to have most optimal values. alpla - Learning rate. 

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch=", epoch + 1, ", loss=", loss)

print("[INFO] evaluating...")
pred_label = predict(test_FV, weight_matrix) # To evalute our model, we use the test fecture vector with a new weights.
print(classification_report(test_label, pred_label))