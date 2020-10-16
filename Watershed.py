import cv2
import numpy as np

kernel = np.ones((3, 3), np.uint8)
colors = np.int32(list(np.ndindex(2, 2, 2))) * 255

img = cv2.imread("./Images/sample (6).jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)

sure_bg = cv2.dilate(blur, kernel, iterations=3)
cv2.imshow('BG', sure_bg)

dist_transform = cv2.distanceTransform(blur, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
cv2.imshow('FG', sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1

markers[unknown == 255] = 0

cv2.watershed(img, markers)
overlay = colors[np.maximum(markers, 0)]
vis = cv2.addWeighted(img, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)

cv2.imshow('Watershed', vis)

cv2.waitKey(0)
cv2.destroyAllWindows()

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html?highlight=watershed
