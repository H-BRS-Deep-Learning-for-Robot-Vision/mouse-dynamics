from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
from os import listdir
from os.path import isfile, join
import glob
import re
import numpy as np
from dateutil import parser
import matplotlib.pyplot as plt
from move_dataset import MoveDataset

dataset = MoveDataset(pickle_file='move_data.pkl')

X,Y = dataset.train_data()

Y = np.argmax(Y,axis=1)

print(X.shape)

X = preprocessing.StandardScaler().fit_transform(X.reshape(-1,256))

print(X.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

print(X_train.shape)
print(Y_train.shape)

clf=RandomForestClassifier(n_estimators=5)
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
print(y_pred)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

