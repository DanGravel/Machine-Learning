import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

#Finds the outermost points from a list of points
def points(pts):
	rectangle = np.zeros((4,2), dtype = "float32")

	#Bottom left point has smallest sum
	#Bottom right point has largest sum
	s = map(sum, pts)
	rectangle[0] = pts[np.argmin(s)]
	rectangle[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rectangle[1] = pts[np.argmin(diff)]
	rectangle[3] = pts[np.argmax(diff)]

	return rectangle

#Fixes perspective to a birds eye view
def perspective_fix(image, pts):
	img = cv2.imread(image)
	rectangle = points(pts)
	(tl, tr, br, bl) = rectangle

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA),int(heightB))

	dst = np.array([
	[0,0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")

	transform_matrix = cv2.getPerspectiveTransform(rectangle,dst)
	warp = cv2.warpPerspective(img,transform_matrix,(maxWidth,maxHeight))

	return warp

#Finds corner coordinates
def corner_coords(image):
	img = cv2.imread(image)
	#Added to remove noise
	img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	dst =cv2.cornerHarris(gray,2,3,0,0.04)

	dst = cv2.dilate(dst, None)

	img[dst>0.01*dst.max()] = [0,0,255]
	#cv2.imshow('dst',img)
	#cv2.waitKey()

	corners = []
	#Creates a list of coordinates
	#Slow, need to find faster way to do this
	for y in range(0, gray.shape[0]):
		for x in range(0, gray.shape[1]):
			c = cv2.cv.Get2D(cv2.cv.fromarray(dst),y,x)
			if c[0] > 0.01*dst.max():
				corners.append((x,y))
	return corners

parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()
img = args.image
cv2.imwrite("perspectiveFix1.jpg",perspective_fix(img, corner_coords(img)))
cv2.waitKey(0)
