import cv2
from skimage.transform import pyramid_gaussian
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help ="Path to img")
ap.add_argument("-s", "--scale", type = float, default = 1.5, help = "scale factor")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

def imagePyramid(image):
    for(i, resized) in enumerate(pyramid_gaussian(image, downscale = 2 )):
        if resized.shape[0] < 50 or resized.shape[1] < 50:
            break
        cv2.imshow("Layer {}".format(i + 1), resized)
        cv2.waitKey(0)

def slidingScanner(image, stepSize, boxSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield(x, y, image[y:y + boxSize[1],x:x + boxSize[0]])
