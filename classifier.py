# Example of building a classification pipeline for the MNIST data set.

from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import time
from csci416 import read_mnist, compute_metrics  # Local utilities.

print('Reading the data in.', end='')
# proper data set, but idk if this is the right way to classify it
X, y, classes = read_mnist('data.csv')
print('Finished Reading Data.')

# Create training and test sets.
test_size_chosen = .8
train_size = 1-test_size_chosen
print('splitting the data...', end='')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size_chosen, random_state=42)
print('done!')
print("Using a training size of " + str(train_size))

# Let's build a data processing pipeline.  This allows us to include scaling and any
# other pre- and post-processing in the object we pickle.
output_file = 'GregsOverfitClassifier.pkl'
clf_pipeline = Pipeline(steps=[('scale', MinMaxScaler()),   # Step 0: apply min-max scaling. # NOTE: This ups it by about 4%
                               ('classify', LinearSVC())])  # Step 1: use a LinearSVC classifier.

print('training the classifier...', end='')
start = time.time()
clf_pipeline.fit(X_train, y_train)
stop = time.time()
print('...done!  elapsed time: {0:f} seconds'.format(stop - start))

print('evaluating the classifier on the test set...')
start = time.time()
compute_metrics(clf_pipeline, X_test, y_test, classes)
stop = time.time()
print('...done!  elapsed time: {0:f} seconds'.format(stop - start))

print('saving the classification pipeline...', end='')
joblib.dump(clf_pipeline, output_file)
print('...done!')
print('classifier written to "{0:s}".'.format(output_file))
