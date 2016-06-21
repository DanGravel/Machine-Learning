import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gzip
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage.io import imread
from PIL import Image

def showImg(image):
	plt.imshow(image , cmap= cm.binary)
	plt.show()

def loadData():
	with gzip.open('mnist.pkl.gz', 'rb') as f:
		data = pickle.load(f , encoding = 'latin1')
	
	#Data is stored in tuple in 3 lists
	X_train, y_train = data[0]
	X_val, y_val = data[1]
	X_test, y_test = data[2]

	#Make the training data into 28x28 matrix
	X_train = X_train.reshape((-1,1,28,28))
	X_val   = X_val.reshape((-1,1,28,28))
	X_test  = X_test.reshape((-1,1,28,28))

	#Set the y values to unsigned integers(uint8)
	y_train = y_train.astype(np.uint8) 
	y_val   = y_val.astype(np.uint8)
	y_test  = y_test.astype(np.uint8)

	return {'X_train':X_train, 'X_val':X_val, 'X_test':X_test,
			'y_train':y_train, 'y_val':y_val, 'y_test':y_test}


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
	print(clf.score(test, loadedData["y_test"]))



trainNetwork()




