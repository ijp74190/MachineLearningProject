#Games data does not include free games

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

games = pd.read_csv('rand_free_games.csv')
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
print("Gaussian Naive Bayes")
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

# Linear Regression
print("Linear Regression")
classifier = LinearRegression()
classifier.fit(X_train, y_train)
y_score = classifier.predict(X_test)
y_pred = []
for num in y_score:
    val = int(round(num,0))
    y_pred.append(0 if val < 0 else val)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test), "\n")

# Logistic Regression
print("Logistic Regression")
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test), "\n")


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
print('accuracy is',accuracy_score(y_pred,y_test))

# KNN
print("KNN, k=9")
classifier = KNeighborsClassifier(n_neighbors=9)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test), "\n")

# Decision Tree's
games = pd.read_csv('games_free_tree.csv')
X = games.iloc[:, :-1].values
y = games.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


print("Decision Tree")
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test))

#Post pruning: best alpha = 0.0007572725453453884
a = 0.0007572725453453884
print("Decision Tree Post Pruning")
classifier = DecisionTreeClassifier(random_state=0, ccp_alpha=a)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test))
'''
path = classifier.cost_complexity_pruning_path(X, y)
alphas, impurities = path.ccp_alphas, path.impurities
param_grid = {'ccp_alpha': alphas}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['ccp_alpha']
print(best_alpha)
clf = DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
print('accuracy is',accuracy_score(y_pred,y_test))
'''
