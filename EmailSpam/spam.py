# dataset source: https://archive.ics.uci.edu/dataset/94/spambase

# import the dataset into my code
from ucimlrepo import fetch_ucirepo

# fetch dataset
spambase = fetch_ucirepo(id=94)

# data 
X = spambase.data.features
Y = spambase.data.targets

# metadata
print(spambase.metadata)

# variable information
print(spambase.variables)

# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#if we upload the dataset by pd.read, split the data into features and target
# X = data.iloc[:, :-1] # all rows, all columns except the last one
# y = data.iloc[:, -1] # all rows, only the last column

# split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# train a logistic regression model
model = LogisticRegression(max_iter=1000) # increasing max_iter for convergence
model.fit(X_train, y_train)

# make prediction on the test set
predictions = model.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

-------------------------------------------------------------------------------
# my result
# Accuracy: 0.9304851556842868
# Confusion Matrix:
#  [[770  34]
#  [ 62 515]]
