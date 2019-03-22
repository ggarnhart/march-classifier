import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from IPython.display import display  # try to fix that display issue
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Local utilities. We're gonna use this to split the data.
from csci416 import read_mnist, compute_metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from classifyByBulk import batch_classify, display_dict_models

data = pd.read_csv('data.csv')
print("let's check the data")
display(data.head())
display(data.describe())  # good up to here!

f = open("data.csv")
data_np = np.loadtxt(f, delimiter=',')  # LETS GOOOO
m, n = data_np.shape
X = data_np[:, 0:6]
y = data_np[:, 7]
classes = np.unique(y)

test_percentage = .7

# At this point, the data should be parse-able.
# i am choosing the random state seed, but idk if i need to
print('data splitting time')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_percentage, random_state=3)
print('data finished splitting!')

# now we have to do the pkl file thing
output_file = 'GregsClassifier.pkl'
# clf_pipline = Pipeline(steps=[('scale', MinMaxScaler())]) TODO: use this after best one is found

# idk what classifier is going to work best, so we'll use all of them...
dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha=1),
    "Naive Bayes": GaussianNB()
}

# Let's test a bunch of them;)


dict_models = batch_classify(
    X_train, y_train, X_test, y_test, no_classifiers=8)
display_dict_models(dict_models)
