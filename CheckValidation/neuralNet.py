import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gzip
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from skimage.io import imread
from sklearn.metrics import accuracy_score

#TODO implement cross validation to determine if overfitting is occuring
def loadData():
	with gzip.open('mnist.pkl.gz', 'rb') as f:
		data = pickle.load(f)

	#Data is stored in tuple in 3 lists
	X_train, y_train = data[0]
	X_validation, y_validation = data[1]
	X_test, y_test = data[2]

	#Make the training data into 28x28 matrix
	X_train = X_train.reshape((-1,1,28,28))
	X_validation   = X_validation.reshape((-1,1,28,28))
	X_test  = X_test.reshape((-1,1,28,28))

	#Set the y values to unsigned integers(uint8)
	y_train = y_train.astype(np.uint8)
	y_validation   = y_validation.astype(np.uint8)
	y_test  = y_test.astype(np.uint8)

	return {'X_train':X_train, 'X_validation':X_validation, 'X_test':X_test,
			'y_train':y_train, 'y_validation':y_validation, 'y_test':y_test}

"""
Accuracy is about 95%
"""
def trainNetwork():
	loadedData = loadData()

	target = loadedData["y_train"]
	features = []
	test = []

	#unrolls the 2d matrix to a vecor to be used as inputs for classifier
	for i in range(loadedData["y_train"].size):
		features.append(loadedData["X_train"][i][0].ravel())
	for i in range(loadedData["y_test"].size):
		test.append(loadedData["X_test"][i][0].ravel())

	clf = RandomForestClassifier(n_estimators = 10, n_jobs = 2)
	clf = clf.fit(features, target)

	predicted = clf.predict(test)
	print(accuracy_score(loadedData["y_test"],predicted))

	return clf

def showImg(image):
	plt.imshow(image , cmap= cm.binary)
	plt.show()

def classifyImg(image, network):
	image = imread("image", flatten = True)
	print(network.predict(image.ravel()))
	print(network.predict_proba(image.ravel()))
	return network.predict(image.ravel())

load = loadData()
img = cv2.imread(["X_test"][0][0])
cv2.imshow('image',img)
cv2.waitKey(0)

"""
net = trainNetwork()
image = imread("eight.png", flatten = True)
print(net.predict(image.reshape(1,-1)))
print(net.predict_proba(image.reshape(1,-1)))
"""
