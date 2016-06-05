import pandas as pd
from sklearn import tree
import numpy as np
"""
Takes a training set the predicts flower type based on measurments
"""
train_url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
iris = pd.read_csv(train_url)

target = list(iris["species"])
feature_one = iris[["sepal_length","sepal_width", "petal_length", "petal_width"]].values
test = [[5.0, 3.5, 1.3, .25 ],
	[6.1, 2.7, 4.2, 1.25],
	[7.2, 3.0, 5.0, 1.9]]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature_one,target)

prediction = clf.predict(test)
print(prediction)



