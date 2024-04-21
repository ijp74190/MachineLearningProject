#Implements ML algorithms for games data

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
import numpy as np

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
svm_acc = accuracy_score(y_pred,y_test)
print('accuracy is',svm_acc,"\n")


# Naive Bayes
# Complement Naive Bayes
print("Complement Naive Bayes")
classifier = ComplementNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
cnb_acc =accuracy_score(y_pred,y_test)
print('accuracy is', cnb_acc)

# Gaussian Naive Bayes
print("Gaussian Naive Bayes")
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
gnb_acc = accuracy_score(y_pred,y_test)
print('accuracy is', gnb_acc)

# Multinomial Naive Bayes
print("Multinomial Naive Bayes")
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
mnb_acc = accuracy_score(y_pred,y_test)
print('accuracy is', mnb_acc)

# Bernoulli Naive Bayes
print("Bernoulli Naive Bayes")
classifier = BernoulliNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
bnb_acc = accuracy_score(y_pred,y_test)
print('accuracy is', bnb_acc, "\n")


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
linreg_acc = accuracy_score(y_pred,y_test)
print('accuracy is',linreg_acc, "\n")


# Logistic Regression
print("Logistic Regression")
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
logreg_acc = accuracy_score(y_pred,y_test)
print('accuracy is', logreg_acc, "\n")


# KNN
print("KNN, k=5")
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
k5_acc = accuracy_score(y_pred,y_test)
print('accuracy is', k5_acc)

# KNN
print("KNN, k=7")
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
k7_acc = accuracy_score(y_pred,y_test)
print('accuracy is', k7_acc)

# KNN
print("KNN, k=9")
classifier = KNeighborsClassifier(n_neighbors=9)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
k9_acc = accuracy_score(y_pred,y_test)
print('accuracy is', k9_acc, "\n")

# KNN
print("KNN, k=11")
classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
k11_acc = accuracy_score(y_pred,y_test)
print('accuracy is', k11_acc, "\n")

# Decision Tree's
games = pd.read_csv('games_tree.csv')
X = games.iloc[:, :-1].values
y = games.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


print("Decision Tree")
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
ptree_acc = accuracy_score(y_pred,y_test)
print('accuracy is', ptree_acc)

#Post pruning: best alpha = 0.001545303351451345
a = 0.001545303351451345
print("Decision Tree Post Pruning")
classifier = DecisionTreeClassifier(random_state=0, ccp_alpha=a)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))
tree_acc = knn_acc = accuracy_score(y_pred,y_test)
print('accuracy is',tree_acc)
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
#Algorithm comparison chart
accuracies = [svm_acc, gnb_acc, linreg_acc, logreg_acc, knn_acc, tree_acc]
algs = ['SVM', 'Naive Bayes', 'Linear Reg.', 'Logistic Reg.',
        'KNN', 'Decision tree']
fig = plt.figure()
plt.bar(algs, accuracies, color='maroon',
        width=0.4)

plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Games data accuracy (adjusted y-axis)")
plt.ylim(.7, .72)
plt.show()

#Algorithm Comparison Chart- no adjust
plt.bar(algs, accuracies, color='maroon',
        width=0.4)

plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Games data accuracy")
plt.ylim(0, 1)
plt.show()



#KNN Comparison
accuracies = [k5_acc, k7_acc, k9_acc, k11_acc]
algs = ['k=5', 'k=7', 'k=9', 'k=11']
# creating the bar plot
plt.bar(algs, accuracies, color='maroon', width=0.4)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("KNN Comparisons (adjusted y-axis)")
plt.ylim(.65, .72)
plt.show()


#Naive Bayes Comparison
accuracies = [cnb_acc, gnb_acc, mnb_acc, bnb_acc]
algs = ['Complement', 'Gaussian', 'Multinomial', 'Bernoulli']
plt.bar(algs, accuracies, color='maroon', width=0.4)
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.title("Naive Bayes Comparisons")
plt.ylim(0, .8)
plt.show()

#Decision Tree Comparison
accuracies = [ptree_acc, tree_acc]
algs = ['No Pruning', 'Post Pruning']
plt.bar(algs, accuracies, color='maroon', width=0.4)
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.title("Decision Tree Comparisons (adjusted y-axis)")
plt.ylim(.65, .72)
plt.show()

'''
y_test = games["Positive_review_threshold"]
print(y_test.value_counts())
print(games.describe())
'''
