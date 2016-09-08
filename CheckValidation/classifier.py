import pickle
import numpy as np
import cv2
import scipy.misc
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.metrics import accuracy_score

def trainNetwork():
	dataset = datasets.fetch_mldata("MNIST Original")
	features = np.array(dataset.data, 'int16')
	target = np.array(dataset.target, 'int')

	hog_features_fd = []

	for feature in features:
		fd = hog(feature.reshape((28,28)),
			orientations = 9,
			pixels_per_cell = (14,14),
			cells_per_block = (1,1),
			visualise = False)
		hog_features_fd.append(fd)
	hog_features = np.array(hog_features_fd, 'float64')

	clf = LinearSVC()
	clf.fit(hog_features, target)

	joblib.dump(clf, "numbers_clf.pkl", compress = 3)


def classifyImg(image):
	clf = joblib.load("Data/numbers_clf.pkl")
	img = cv2.imread(image)

	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#imgGray = cv2.GaussianBlur(imgGray, (5,5), 0)

	#ret, thresh = cv2.threshold(imgGray,100, 255,cv2.THRESH_BINARY_INV)
	#roi = thresh
	#cv2.imshow("roi", roi)
	#cv2.waitKey(0)
	#roi = cv2.dilate(roi, (3,3))
	#cv2.imshow("roi", roi)
	#cv2.waitKey(0)
 	hog_feature = hog(imgGray,
		orientations = 9,
		pixels_per_cell = (14,14),
		cells_per_block = (1,1),
		visualise = False)
	number = clf.predict(np.array([hog_feature]))
	print(number)
	#print(clf.decision_function(np.array([hog_feature])))
	return number
