from sklearn import metrics
import numpy as np

# Report various performance metrics for a classifier.
def compute_metrics(classifier, X_test, y_test, classes):

    # Use the classifier to make predictions for the test set.
    y_pred = classifier.predict(X_test)

    # Print the class names.
    print('Classes:', classes, '\n')

    # Report the accuracy.
    print('Accuracy: {0:.2f} %'.format(metrics.accuracy_score(y_test, y_pred)))

    # Compute the confusion matrix.
    cm = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion matrix, without normalization')
    print(cm, '\n')

    # Normalize the confusion matrix by row (i.e by the number of samples in each class).
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=3, linewidth=132)
    print('Normalized confusion matrix')
    print(cm_normalized, '\n')

    # The confusion matrix as percentages.
    cm_percentage = 100 * cm_normalized
    print('Confusion matrix as percentages')
    print(np.array2string(cm_percentage, formatter={'float_kind':lambda x: "%6.2f" % x}), '\n')

    # Precision, recall, and f-score.
    print(metrics.classification_report(y_test, y_pred))

# Read the MNIST data set.
def read_mnist (file):
    Z = np.load(file)
    m, n = Z.shape
    X = np.float64(Z[:, 0:n-1])
    y = Z[:, n-1]
    classes = np.unique(y)
    return X, y, classes
