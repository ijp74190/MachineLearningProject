import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#print(confusion_matrix(y_test, y_pred))

games = pd.read_csv('rand_games.csv')
X = games.iloc[:, :-1].values
y = games.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#SVM
print("Linear SVC (SVM)")
svc = LinearSVC(dual=False)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
print("svm accuracy of training data=", acc_svc)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test),"\n")


# Naive Bayes
# Complement Naive Bayes
print("Complement Naive Bayes")
classifier = ComplementNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test))

# Gaussian Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test))

# Multinomial Naive Bayes
print("Multinomial Naive Bayes")
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test))

# Bernoulli Naive Bayes
print("Bernoulli Naive Bayes")
classifier = BernoulliNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test), "\n")

'''
NOT A BINARY CLASSIFIER

# Linear Regression
print("Linear Regression")
classifier = LinearRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test), "\n")
'''

# Logistic Regression
print("Logistic Regression")
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test), "\n")


# KNN
print("KNN, k=3")
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test))

# KNN
print("KNN, k=5")
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test))

# KNN
print("KNN, k=7")
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test), "\n")

# Decision Tree's
print("Decision Tree")
classifier = DecisionTreeClassifier(max_leaf_nodes=10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test))