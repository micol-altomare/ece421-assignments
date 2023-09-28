# ECE421 Assignment 3 Question 2

import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# Premable
iris = load_iris()              # 150 entries
iris_hundred = iris.data[:100]  # only the first 100 entries


# Selecting only the first two features (sepal length and width)
X = iris_hundred[:, :2]
y = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(iris_hundred[:, :2], iris.target[:100], test_size=0.8, random_state=0)



# Question 2.1
'''
(2 points) Implement a binary linear classifier on the first two dimensions (sepal 
length and width) of the iris dataset and plot its decision boundary. (Hint: sklearn 
refers to the binary linear classifier as a LogisticRegression, we will see why 
later in the course.)
'''

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Plot the decision boundary
# Create a meshgrid for the plot
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the labels for each point in the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Reshape the predicted labels to match the shape of the meshgrid
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
# TODO: add legend
plt.title('Binary Linear Classifier Decision Boundary') # TODO: improve title
plt.show() # TODO: when I zoom in on the deicision boundary, it looks like it's not a straight line + multiple colours. Why is that?



# Question 2.2
'''
(1 point) Report the accuracy of your binary linear classifier on both the 
training and test sets.
'''
training_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print('Training Accuracy: ', training_accuracy) # TODO: why is it 100%?
print('Test Accuracy: ', test_accuracy)