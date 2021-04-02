import cv2
import numpy as np
img = cv2.imread("./Images/sample (4).jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = np.float32(gray_img)

corner_img = cv2.cornerHarris(gray_img, 3, 9, 0.04)
corner_img = cv2.dilate(corner_img, None)
img[corner_img > 0.01 * corner_img.max()] = [0, 0, 255]
cv2.imshow('Corner Image', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
    