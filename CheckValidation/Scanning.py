import cv2
from skimage.transform import pyramid_gaussian
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(boxW, boxH) = (28,28) #Size of training data

def imagePyramid(image, downScale):
    for(i, resized) in enumerate(pyramid_gaussian(image, downscale)):
        if resized.shape[0] < 50 or resized.shape[1] < 50:
            break
        yield(resized)


def slidingScanner(image, stepSize, boxSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield(x, y, image[y:y + boxSize[1],x:x + boxSize[0]])

def scanImg(image, stepSize, boxSize, downScale)
for resized in imagePyramid(image, downscale = 1.5):
    for(x,y, window) in slidingScanner(image, stepSize = 100, boxSize = (boxW, boxH)):
        if window.shape[0] != boxH or window.shape[1] != boxW:
            continue

                #add neuralnet

        clone = resized.copy()
        cv2.rectangle(clone,(x,y), (x + boxW, y + boxH), (0,255,0),2)
        cv2.imshow("window",clone)
        cv2.waitKey(1)
        time.sleep(0.025)
