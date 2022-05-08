import cv2
import numpy as np

c = cv2.imread("../Datasets/Training/car/car (2).jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", c)

# _, th1 = cv2.threshold(c, 100, 255, cv2.THRESH_BINARY)  # Binary Threshold from b&w color img.
th0 = cv2.adaptiveThreshold(
    c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
th1 = cv2.adaptiveThreshold(
    c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 50)
th2 = cv2.adaptiveThreshold(
    c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
th3 = cv2.adaptiveThreshold(
    c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 50)


cv2.imshow('Threshold 0, 0', th0)
cv2.imshow('Threshold 0, 50', th1)
cv2.imshow('Threshold 10, 0', th2)
cv2.imshow('Threshold 10, 50', th3)

cv2.waitKey(0)
cv2.destroyAllWindows()
