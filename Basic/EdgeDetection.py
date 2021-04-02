import cv2
import numpy as np
img = cv2.imread("/Datasets/Training/Cars/sample (6).jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imshow("Original ", img)

gaussian_img = cv2.GaussianBlur(img, (11, 11), 0)
sobelx_img = cv2.Sobel(gaussian_img, cv2.CV_8U, 1, 0, ksize=-1)
cv2.imshow("Sobel X", sobelx_img)

sobely_img = cv2.Sobel(gaussian_img, cv2.CV_8U, 0, 1, ksize=-1)
cv2.imshow("Sobel Y", sobely_img)

# Combinig both by normalising it.
edgeXY = np.sqrt(sobelx_img ** 2 + sobely_img ** 2)
edgeXY = np.uint8(edgeXY)
cv2.imshow("Sobel XY", edgeXY)

canny_img = cv2.Canny(gaussian_img, 60, 70)
cv2.imshow("Canny", canny_img)

cv2.waitKey(0)
cv2.destroyAllWindow()
