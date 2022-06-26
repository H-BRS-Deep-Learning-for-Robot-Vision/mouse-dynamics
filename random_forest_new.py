from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
from move_dataset import MoveDataset
from keras_utils import truncate_model, get_model_output_features
import tensorflow.keras as keras
import tensorflow as tf

dataset = MoveDataset(pickle_file='move_data.pkl')

X, Y = dataset.train_data()

Y = np.argmax(Y, axis=1)
X = preprocessing.StandardScaler().fit_transform(X.reshape(-1, 256)).reshape(-1, 128, 2)
model = truncate_model(tf.keras.models.load_model('final_model.h5'))

# Do a split of training and test data with a 80%/20% split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_new = get_model_output_features(model, X_train)
X_test_new = get_model_output_features(model, X_test)

all_accuracies = []

for userid in range(dataset.unique_user_count()):
    # Create a y vector for which the value is 1 if it is equal to the userid or 0 otherwise.
    y_train = np.zeros(len(X_train_new))
    y_train[np.where(Y_train == userid)[0]] = 1

    y_test = np.zeros(len(X_test_new))
    y_test[np.where(Y_test == userid)[0]] = 1

    clf = RandomForestClassifier(n_estimators=64, n_jobs=-1)
    clf.fit(X_train_new, y_train)
    y_pred = clf.predict(X_test_new)
    all_accuracies.append(metrics.accuracy_score(y_test, y_pred))
    print("Accuracy:", all_accuracies[-1])

print("Mean accuracy:", np.mean(all_accuracies))


