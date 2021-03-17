import cv2
import numpy as np


img1_path = ".\\Images\\sample (3).jpg"

c = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
c = cv2.resize(c, None, fx=0.5, fy=0.5)
cv2.imshow("Original", c)

_, th1 = cv2.threshold(c, 100, 255, cv2.THRESH_BINARY)  # Binary Threshold from b&w color img.
# th2 = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('Threshold', th1)

cv2.waitKey(0)
cv2.destroyAllWindows()
