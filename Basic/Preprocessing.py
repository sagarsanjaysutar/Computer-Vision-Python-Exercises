import imutils
import cv2

'''
    To resize and maintain aspect ratio.
'''
image = cv2.imread('../Datasets/Testing/cat/cat.1676.jpg')
(h, w) = image.shape[:2]
dW, dH = 0, 0 # Delta width and height

cv2.imshow('Original Image', image)
print("Original dimentions: ", image.shape, dH, dW)

if h > w:
    # Crop the height and resize alongside the width
    image = imutils.resize(image, width = 256, inter = cv2.INTER_AREA)

    # We take the difference between larger dim and desired dim.
    # We then divde it by 2 in order to ignore pixel values on both the sides along axis.
    dH = int((image.shape[0] - 256) / 2) 
    cv2.imshow('Reside along width', image)
else:
    # Crop the width and resize alongside the height
    image = imutils.resize(image, height = 256, inter = cv2.INTER_AREA)
    dW = int((image.shape[1] - 256) / 2)
    cv2.imshow('Resized along height', image)

print("Dimentions after resizing along shorter dimension ", image.shape, dH, dW)
(h, w) = image.shape[:2]

''' 
Cropping
    roi = image[startY:endY, startX:endX]
    cropped_image = image[dH:h - dH, dW:w - dW]
    The startY and startX are dH and dW respectively. 
    By starting from delta offset i.e. [dH:h, dW:w], we ignore pixels from 0, 0 to dH, hW.
    By substracting the delta offset i.e [dH:h - dH, dW:w - dW], we ignore dH, dW pixels from the end.
'''
image = image[dH:h - dH, dW:w - dW]
image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
print("Dimentions after resizing along shorter dimension and cropping along larger dimension ", image.shape, dH, dW)
cv2.imshow('Cropped Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
