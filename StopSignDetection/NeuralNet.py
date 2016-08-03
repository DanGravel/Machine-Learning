import os
import glob
import cv2
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

target = []
images = []
for file in os.listdir("Stop Signs"):
    img = cv2.imread(os.path.join("Stop Signs",file))
    if img is not None:
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if file[0] == '5':
            print("yield %s", file[0])
            target.append(0)
        else:
            target.append(1)
            print("stop %s", file[0])
        cv2.imshow("tset",img)
        cv2.waitKey(0)

def trainNetwork():

	features = []

	#unrolls the 2d matrix to a vecor to be used as inputs for classifier
	for i in range(len(target)):
		features.append(images[i].ravel())

	clf = RandomForestClassifier(n_estimators = 10, n_jobs = 2)
	clf = clf.fit(features, target)

	return clf


clf =  trainNetwork()
