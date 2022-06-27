import argparse

from sklearn.metrics import ConfusionMatrixDisplay

from move_dataset import MoveDataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from keras_utils import get_model_output_features, truncate_model
from sklearn.svm import OneClassSVM
from sklearn import metrics
import numpy as np
from random import uniform
import pandas as pd
import matplotlib.pyplot as plt

dataset = MoveDataset(pickle_file='move_data.pkl')

X, y_onehot = dataset.train_data()
X = preprocessing.StandardScaler().fit_transform(X.reshape(-1, 256)).reshape(-1, 128, 2)

# One hot vector to single value vector
y = np.argmax(y_onehot, axis=1)

# Do a split of training and test data with a 80%/20% split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)


# Source: https://github.com/margitantal68/sapimouse/blob/40b5ea6cf10c6f1d64b9dd0427d21138cc4f75e2/util/oneclass.py#L40
def compute_AUC_EER(positive_scores, negative_scores):
    zeros = np.zeros(len(negative_scores))
    ones = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    EER_fpr = fpr[np.argmin(np.absolute((fnr - fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr - fpr)))]
    EER = 0.5 * (EER_fpr + EER_fnr)
    return roc_auc, EER


# Source: https://github.com/margitantal68/sapimouse/blob/40b5ea6cf10c6f1d64b9dd0427d21138cc4f75e2/util/oneclass.py#L138
def score_normalization(positive_scores, negative_scores):
    scores = [positive_scores, negative_scores]
    scores_df = pd.DataFrame(scores)

    mean = scores_df.mean()
    std = scores_df.std()
    min_score = mean - 2 * std
    max_score = mean + 2 * std

    min_score = min_score[0]
    max_score = max_score[0]

    positive_scores = [(x - min_score) / (max_score - min_score) for x in positive_scores]
    positive_scores = [(uniform(0.0, 0.05) if x < 0 else x) for x in positive_scores]
    positive_scores = [(uniform(0.95, 1.0) if x > 1 else x) for x in positive_scores]

    negative_scores = [(x - min_score) / (max_score - min_score) for x in negative_scores]
    negative_scores = [uniform(0.0, 0.05) if x < 0 else x for x in negative_scores]
    negative_scores = [uniform(0.95, 1.0) if x > 1 else x for x in negative_scores]
    return positive_scores, negative_scores


all_accuracies = []
total_confusion_matrix = np.zeros((2, 2))

total_y_test = np.empty(0)
total_y_pred = np.empty(0)

if __name__ == "__main__":
    # Extract parameters from command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model_trained.h5")
    args = parser.parse_args()

    model = truncate_model(tf.keras.models.load_model(args.model))

    for userid in range(dataset.unique_user_count()):
        if np.count_nonzero(Y_train == userid) == 0:
            continue
        print('User: {}/{}'.format(userid, dataset.unique_user_count()))
        X_positive = X_train[Y_train == userid]
        X_negative = X_train[Y_train != userid]

        # Extract new features from dx, dy vectors
        X_positive_features = get_model_output_features(model, X_positive)
        X_negative_features = get_model_output_features(model, X_negative)

        clf = OneClassSVM(gamma='scale')
        clf.fit(X_positive_features)

        X_positive_test = X_test[Y_test == userid]
        X_negative_test = X_test[Y_test != userid]

        if X_positive_test.shape[0] > 0 and X_negative_test.shape[0] > 0:
            X_positive_test = get_model_output_features(model, X_positive_test)
            X_negative_test = get_model_output_features(model, X_negative_test)

            y_pred_positive = clf.predict(X_positive_test)
            y_pred_negative = clf.predict(X_negative_test)

            y_test = np.empty(len(y_pred_positive) + len(y_pred_negative))
            y_test[:len(y_pred_positive)] = 1
            y_test[len(y_pred_positive):] = -1
            y_pred_total = np.append(y_pred_positive, y_pred_negative)
            all_accuracies.append(metrics.accuracy_score(y_test, y_pred_total))
            print(all_accuracies[-1])

            total_y_test = np.append(total_y_test, y_test)
            total_y_pred = np.append(total_y_pred, y_pred_total)

        # positive_scores = clf.score_samples(X_positive)
        # negative_scores = clf.score_samples(X_negative)

        # auc, eer = compute_AUC_EER(positive_scores, negative_scores)

        # positive_scores, negative_scores = score_normalization(positive_scores, negative_scores)
        # print('auc:', auc, 'eer:', eer)

# print mean accuracy
print('mean accuracy:', np.mean(all_accuracies))

ConfusionMatrixDisplay.from_predictions(total_y_test, total_y_pred,
                                        display_labels=['Actual User', 'Impostor']).plot()
plt.show()
