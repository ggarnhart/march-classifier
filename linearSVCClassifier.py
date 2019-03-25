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

from sklearn.model_selection import RandomizedSearchCV

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

# try just this, first.
clf_pipeline = Pipeline(
    steps=[('scale', MinMaxScaler()), ('classify', LinearSVC())])
start = time.time()
clf_pipeline.fit(X_train, y_train)
stop = time.time()

start = time.time()
compute_metrics(clf_pipeline, X_test, y_test, classes)
stop = time.time()

print('saving the classification pipeline...', end='')
joblib.dump(clf_pipeline, output_file)
print('...done!')
print('classifier written to "{0:s}".'.format(output_file))

# let's set up our linear svc set
penalty = ['l1', 'l2']
loss = ['hinge', 'squared_hinge']
dual = [True, False]
tol = [1e-4, .12, 1e-2, 1e-1, 1]
C = [1, .5, 1.5]
multi_class = ['ovr', 'crammer_singer']
fit_intercept = [True, False]
intercept_scaling = [1, .5, 1.5, 2]
verbose = [0, 1, 2]
random_state = [None, 23, 1]
# max_iter = [1000, 100, 10000, 1]

random_grid = {'penalty': penalty, 'loss': loss, 'dual': dual,
               'tol': tol, 'C': C, 'multi_class': multi_class, 'fit_intercept': fit_intercept, 'intercept_scaling': intercept_scaling, 'verbose': verbose, 'random_state': random_state}
# thing we are going to be using (in this case, a LinearSVC)
svc = LinearSVC()
svc_random = RandomizedSearchCV(estimator=svc, param_distributions=random_grid,
                                n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

svc_random.fit(X_train, y_train)


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


# build a model we can compare our randomized search with
base_svc = LinearSVC()
base_svc.fit(X_train, y_train)
base_svc_accuracy = evaluate(base_svc, X_test, y_test)

best_random = svc_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)

print('Improvement of {:0.2f}%.'.format(
    100 * (random_accuracy - base_svc_accuracy) / base_svc_accuracy))
