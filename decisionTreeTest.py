import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from IPython.display import display  # try to fix that display issue
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.externals import joblib  # used to export the classifier

# Local utilities. We're gonna use this to split the data.
from csci416 import read_mnist, compute_metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from classifyByBulk import batch_classify, display_dict_models

from sklearn.tree import DecisionTreeClassifier
# we don't know what to do, so let's try everything.
print()
print()
data = pd.read_csv('data.csv')
# print("let's check the data")
# display(data.head())
# display(data.describe())  # These two show useful data things, but after you've run this 100 times, you don't really need to see it

f = open("data.csv")
data_np = np.loadtxt(f, delimiter=',')
m, n = data_np.shape
X = data_np[:, 0:6]
y = data_np[:, 7]
classes = np.unique(y)

test_percentage = .7

# At this point, the data should be parse-able.
# i am choosing the random state seed, but idk if i need to
print('data splitting time')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_percentage, random_state=31)
print('data finished splitting!')

# now we have to do the pkl file thing
output_file = 'GregsClassifier.pkl'
# clf_pipline = Pipeline(steps=[('scale', MinMaxScaler())]) TODO: use this after best one is found

# dict_models = batch_classify(X_train, y_train, X_test, y_test)
# display_dict_models(dict_models)

# Above two calls allowed us to bulk test 8 classifiers. They all sucked
# We're gonna try linear SVC

# try just this, first.
clf_pipeline = Pipeline(
    steps=[('classify', DecisionTreeClassifier(max_depth=5, min_samples_leaf=3))])  # decision tree = 34 %
# print('training the SVC..', end='')
start = time.time()
clf_pipeline.fit(X_train, y_train)
stop = time.time()
# print('...done!  elapsed time: {0:f} seconds'.format(stop - start))

# print('evaluating the classifier on the test set...')
start = time.time()
compute_metrics(clf_pipeline, X_test, y_test, classes)
stop = time.time()
# print('...done!  elapsed time: {0:f} seconds'.format(stop - start))

# print('saving the classification pipeline...', end='')
joblib.dump(clf_pipeline, output_file)
# print('...done!')
# print('classifier written to "{0:s}".'.format(output_file))
