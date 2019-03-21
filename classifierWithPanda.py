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

data = pd.read_csv('data.csv')
print("let's check the data")
display(data.head())
display(data.describe())  # good up to here!

f = open("data.csv")
data_np = np.loadtxt(f, delimiter=',')  # LETS GOOOO
m, n = data_np.shape
X = data_np[:, 0:6]
Y = data_np[:, 7]

# # Let's use the nice csci416 function to split the data pls.
# test_size_amount = .7
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=test_size_amount, random_state=42)
