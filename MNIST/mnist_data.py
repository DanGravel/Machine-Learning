import matplotlib as mpl
import numpy as np
import gzip
import sys
from struct import unpack
from matplotlib import pyplot
from numpy import zeros, uint8
from pylab import imshow, show, cm
def getData(imageFile, labelFile):

	images = gzip.open(imageFile, 'rb')
	labels = gzip.open(labelFile, 'rb')
	
	#Data for images

	#When reading the 4 is required as unpack requires a bytes object of length 4
	images.read(4)
	numImgs = images.read(4)
	#Need big endian unsigned int so '>I' is required
	numImgs = unpack('>I', numImgs)[0]
	rows = images.read(4)
	rows = unpack('>I', rows)[0]
	cols = images.read(4)
	cols = unpack('>I', cols)[0]
	
	#Data for labels
	labels.read(4)
	N = labels.read(4)
	N = unpack('>I', N)[0]

	x = zeros((N,rows,cols), dtype = uint8)
	y = zeros((N,1), dtype = uint8)
	#put data in x and y
	for i in range(N):
		for row in range(rows):
			for col in range(cols):
				pixl = images.read(1)
				pixl = unpack('>B', pixl)[0]
				x[i][row][col] = pixl

		tmp_label = labels.read(1)
		y[i] = unpack('>B', tmp_label)[0] 
	return(x,y)
	

def showImg(image):
    imshow(image, cmap=cm.gray)
    show()


