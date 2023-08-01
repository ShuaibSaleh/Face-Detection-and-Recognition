import numpy as np
import cv2


# this function takes the image data that sent from js code and new image name
# then saves the image to the input folder and returns its path
def saveImage(imgData, imgName):
    path = f'./static/images/input/{imgName}.jpg'
    with open(path, 'wb') as f:
        f.write(imgData)

    return path


# this function takes the image path
# then reads the image as grayscale image and resize it and returns the result
def readImg(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (600, 600))
    return img
