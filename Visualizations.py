#Visualizations for game data

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
from matplotlib import colors

#print(confusion_matrix(y_test, y_pred))

games = pd.read_csv('rand_games.csv')
X = games.iloc[:, :-1].values
y = games.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


met = games['Metacritic_score']
med = games['Median_playtime']
pri = games['Price']

# Create the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

my_cmap = plt.get_cmap('hsv')

# Plot the values
chart = ax.scatter(met, pri, med, c = med, marker='.', cmap=my_cmap)
ax.set_xlabel('Metcritic Score')
ax.set_ylabel('Price')
ax.set_zlabel('Median Playtime')
plt.title("Games Data")
fig.colorbar(chart, ax = ax, shrink = 0.5, aspect = 5)
plt.show()



#Training Data
met = X_train[  : ,0]
med = X_train[  : ,1]
pri = X_train[  : ,2]

# Create the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
my_cmap = plt.get_cmap('hsv')

# Plot the values
chart = ax.scatter(met, pri, med, c = med, marker='.', cmap=my_cmap)
ax.set_xlabel('Metacritic Score')
ax.set_ylabel('Price')
ax.set_zlabel('Median Playtime')
plt.title("Training Data")
fig.colorbar(chart, ax = ax, shrink = 0.5, aspect = 5)
plt.show()



#Testing Data
met = X_test[  : ,0]
med = X_test[  : ,1]
pri = X_test[  : ,2]

# Create the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
my_cmap = plt.get_cmap('hsv')

# Plot the values
chart = ax.scatter(met, pri, med, c = med, marker='.', cmap=my_cmap)
ax.set_xlabel('Metacritic Score')
ax.set_ylabel('Price')
ax.set_zlabel('Median Playtime')
plt.title("Testing Data")
fig.colorbar(chart, ax = ax, shrink = 0.5, aspect = 5)
plt.show()



#Whole Data
met = games['Metacritic_score']
med = games['Median_playtime']
pri = games['Price']
label = games['Positive_review_threshold']
color = ['red','green']

# Create the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the values
ax.scatter(met, pri, med, c=label, cmap=colors.ListedColormap(color), marker='.')
ax.set_xlabel('Metacritic Score')
ax.set_ylabel('Price')
ax.set_zlabel('Median Playtime')
plt.title("Games Data")
plt.show()


#Training Data
met = X_train[  : ,0]
med = X_train[  : ,1]
pri = X_train[  : ,2]
label = y_train
color = ['red','green']

# Create the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the values
ax.scatter(met, pri, med, c=label, cmap=colors.ListedColormap(color), marker='.')
ax.set_xlabel('Metacritic Score')
ax.set_ylabel('Price')
ax.set_zlabel('Median Playtime')
plt.title("Training Data")
plt.show()



#Testing Data
met = X_test[  : ,0]
med = X_test[  : ,1]
pri = X_test[  : ,2]
label = y_test
color = ['red','green']

# Create the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the values
ax.scatter(met, pri, med, c=label, cmap=colors.ListedColormap(color), marker='.')
ax.set_xlabel('Metacritic Score')
ax.set_ylabel('Price')
ax.set_zlabel('Median Playtime')
plt.title("Testing Data")
plt.show()

#Real, predicted charts (maybe third color for misclassed)
