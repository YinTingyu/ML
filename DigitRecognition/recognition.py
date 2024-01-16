# import dataset by sklearn
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
# %matplotlib inline
# if load dataset by tensorflow
# from tensorflow.keras.datasets import mnist

mnist = load_digits()
type(mnist)
mnist.keys()

# import neccessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.25, random_state=42)
print('Training data and target sizes: \n{}, {}'.format(X_train.shape, y_train.shape))
print('Test data and target sizes: \n{}, {}'.format(X_test.shape, y_test.shape))

# train and test by tree
from sklearn import tree
class_tree=tree.DecisionTreeClassifier()
class_tree.fit(X_train, y_train)
predictions = class_tree.predict(X_test)

print('Accuracy of model = %2f%%' % (accuracy_score(y_test, predictions)*100))

# train and test by Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# input data normalization
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# try MLPClassifier
mlp = MLPClassifier()
mlp.fit(X_train_scaled, y_train)
nn_predictions = mlp.predict(X_test_scaled)

print('Accuracy of model = %2f%%' % (accuracy_score(y_test, nn_predictions)*100))

# try Multinomial naive bayes classifier
from sklearn.naive_bayes import MultinomialNB

naive_classifier = MultinomialNB()
naive_classifier.fit(X_train, y_train)
mnb_predictions = naive_classifier.predict(X_test)

print('Accuracy of model = %2f%%' % (accuracy_score(y_test, mnb_predictions)*100))

# try K-Neareast Neighbors classifier
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

print("Accuracy of model = %2f%%" % (accuracy_score(y_test, knn_predictions)*100))

# try Support Vector Machine(SVM)
from sklearn import svm

svm_classifier = svm.SVC(gamma=0.001)
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)

print("Accuracy of model = %2f%%" % (accuracy_score(y_test, svm_predictions)*100))

# make a report for the predictions of the classifier
from sklearn import metrics
print("Classification report for SVM classifier %s:\n%\n" % (knn, metrics.classification_report(y_test, knn_predictions)))

# confusion matrix
print("Confustion matrix:\n%s" % metrics.confustion_matrix(y_test, knn_predictions))








