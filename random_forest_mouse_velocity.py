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

user_path = 'DLRV_Dataset/user_df_normalized_speed'
actions = ['scroll','move']
scroll = pd.DataFrame()
move = pd.DataFrame()

for action in actions:
    path = listdir(user_path+'/'+action)
    for file in path:
        df = pd.read_pickle(user_path+'/'+action+'/'+file)
        if action == 'move':
            move = pd.concat([move,df])
        if action == 'scroll':
            scroll = pd.concat([scroll,df])

move.info()
# scroll.info()

data = move[['dx/dt','dy/dt','token']].dropna()
X_move = data[['dx/dt','dy/dt']]
Y_move = data[['token']]
# Y_move.info()
# X_move.info()

#Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_move, Y_move, test_size=0.3) # 70% training and 30% test

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=64,n_jobs=-1)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Move Model Accuracy:",metrics.accuracy_score(y_test, y_pred))