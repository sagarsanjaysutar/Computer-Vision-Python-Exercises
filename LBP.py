from skimage import feature
import numpy as np
from sklearn.svm import LinearSVC
from imutils import paths
import cv2
import os

imageInfo = {
    "training": "Images\\Training",
    "testing": "Images\\Testing"
}
data = []
labels = []

def describe(numPoints, radius, image, eps=1e-7):
    lbp = feature.local_binary_pattern(image, numPoints, radius, 'uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    hist = hist.astype('float')
    hist /= (hist.sum() + eps)
    return hist

for imagePath in paths.list_images(imageInfo["training"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = describe(24, 8, gray)
    # imagePath = Images / training / keyboard / keyboard_01.jpeg . Split it with "/" and we get - 2 elemet as the name i.e Folder name    
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)

model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

for imagePath in paths.list_images(imageInfo["testing"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = describe(24, 8, gray)
    predectition = model.predict(hist.reshape(1, -1)) #convert into 2d
    cv2.putText(image, predectition[0], (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)
    cv2.imshow("LBP", image)
    cv2.waitKey(0)
