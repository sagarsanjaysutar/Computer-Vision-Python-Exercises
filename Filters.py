import cv2

img = cv2.imread("./Images/sample (1).jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
dim = "Width " + str(img.shape[1]) + " Height " + str(img.shape[0])
cv2.putText(img, dim, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_4)
cv2.imshow("Original ", img)

img = cv2.GaussianBlur(img, (15, 15), 5000)
cv2.imshow("Gaussian blur - Sigma 5000", img)

img = cv2.GaussianBlur(img, (15, 15), 0)
cv2.imshow("Gaussian blur - Sigma 0 ", img)

img = cv2.medianBlur(img, 5)
cv2.imshow("Median blur - Ksize 5", img)

img = cv2.medianBlur(img, 9)
cv2.imshow("Median blur - Ksize 9", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
