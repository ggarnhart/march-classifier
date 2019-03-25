# LARGELY, this experiement failed.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from IPython.display import display  # try to fix that display issue
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# we don't know what to do, so let's try everything.

data = pd.read_csv('data.csv')
print("Starting")


f = open("data.csv")
data_np = np.loadtxt(f, delimiter=',')
m, n = data_np.shape
X = data_np[:, 0:6]
y = data_np[:, 7]
classes = np.unique(y)

test_percentage = .7

# At this point, the data should be parse-able.
# i am choosing the random state seed, but idk if i need to
# print('data splitting time')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_percentage, random_state=31)
# print('data finished splitting!')

# now we have to do the pkl file thing
output_file = 'GregsClassifier.pkl'

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
