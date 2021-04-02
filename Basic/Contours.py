import cv2
import numpy as np
gray, blur, edge, contours, cont_img = list(), list(), list(), list(), list()
img = [cv2.imread("/Datasets/Training/Keyboard/keyboard (1).jpg"),
       cv2.imread("/Datasets/Training/Keyboard/keyboard (2).jpg"),
       cv2.imread("/Datasets/Training/Keyboard/keyboard (3).jpg"),
       cv2.imread("/Datasets/Training/Keyboard/keyboard (4).jpg")
    ]
for i in range(len(img)):
    gray.append(cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY))
    blur.append(cv2.GaussianBlur(gray[i], (11, 11), 0))
    edge.append(cv2.Canny(blur[i], 60, 70))
    cnts, hierarchy = cv2.findContours(edge[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.append(cnts)
    cont_img.append(cv2.drawContours(blur[i], contours[i], -1, (0, 255, 255), 3))
    cv2.imshow("Contours ", cont_img[i])
    cv2.waitKey(0)
cv2.destroyAllWindows()
