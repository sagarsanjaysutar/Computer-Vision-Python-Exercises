from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skimage import io, color, feature, transform

mnist = datasets.load_digits()
images = mnist.images
data_size = len(images)

# Preprossesing the images
images = images.reshape(len(images), -1)
labels = mnist.target

LR_classifier = LogisticRegression(C=0.01, penalty='l2', tol=0.01, max_iter=2000)
LR_classifier.fit(images[: int((data_size / 4) * 3)], labels[: int((data_size / 4) * 3)])

sample_img = io.imread("../Datasets/Training/Digits/nine.png")
sample_img = color.rgb2gray(sample_img)
sample_img = transform.resize(sample_img, (8, 8), mode="wrap")
sample_edge = feature.canny(sample_img, sigma=5)
sample_edge = sample_edge.flatten()
prediction = LR_classifier.predict(sample_edge)
print(prediction)
