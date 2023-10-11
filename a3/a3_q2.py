# ECE421 Assignment 3 Question 2

import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



# Preamble
iris = load_iris()              # 150 entries in total
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

# Create a meshgrid for the plot
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1.2, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print(X[:, 1].min() - 1)
# Predict the labels for each point in the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Reshape the predicted labels to match the shape of the meshgrid
Z = Z.reshape(xx.shape)

# Plot the training points
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

b_bin = model.intercept_    # Bias:          [-5.9685275]
w_bin = model.coef_[0]      # Weight vector: [ 1.80226162 -1.24492959]
a = -w_bin[0] / w_bin[1]
xx_boundary = np.linspace(3.9, 7)
yy_boundary = a * xx_boundary - (b_bin[0]) / w_bin[1]
plt.plot(xx_boundary, yy_boundary, 'k-', c='violet', linewidth=2.5, label='Decision Boundary')

# print weights and bias
print('Weight vector of binary classifier: ', w_bin) # [ 1.80226162 -1.24492959]
print('Bias of binary classifier: ', b_bin) # [-5.9685275]

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Binary Linear Classifier') 
plt.legend(loc='upper right')
plt.show() 



# Question 2.2
'''
(1 point) Report the accuracy of your binary linear classifier on both the 
training and test sets.
'''
training_accuracy = model.score(X_train, y_train) 
test_accuracy = model.score(X_test, y_test)
print('Training Accuracy: ', training_accuracy) # 1.0
print('Test Accuracy: ', test_accuracy)         # 0.9875

# Question 2.3
'''
(2 points) Implement a linear SVM classifier on the first two dimensions (sepal 
length and width). Plot the decision boundary of the classifier and its margins.
'''

# Create a linear SVM classifier
svm_classifier = SVC(kernel='linear', C=1000) 
svm_classifier.fit(X_train, y_train)

# Plot the decision boundary and margins
plt.figure(figsize=(8, 6))

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k', marker='o', s=100, label='Training Points')

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])

# Plot decision boundary and margins
Z = Z.reshape(xx.shape)
contour = plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'], label='_nolegend_')

# Plot support vectors
support_vectors = plt.scatter(svm_classifier.support_vectors_[:, 0], svm_classifier.support_vectors_[:, 1], s=200, facecolors='none',
            edgecolors='g', marker='o', linewidth = 1.5, label='Support Vectors')

w_first = svm_classifier.coef_[0]
b_first = svm_classifier.intercept_
print('Weight vector of SVM with test_size=0.8: ', w_first) # [ 3.33266363 -3.33342658]
print('Bias of SVM with test_size=0.8: ', b_first) # [-7.66277845]


legend_labels = ['Decision Boundary', 'Margins']

legend_handles = [
    Line2D([0], [0], color='k', linestyle='-', label=legend_labels[0]),
    Line2D([0], [0], color='k', linestyle='--', label=legend_labels[1]),
    #Line2D([0], [0], color='k', linestyle='--', label=legend_labels[2]),
    support_vectors
]

plt.legend(handles=legend_handles)

plt.title('Linear SVM Classifier on Sepal Length and Width, test_size=0.8')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()


# Question 2.4
'''
(1 point) Circle the support vectors. Please justify how to identify them 
through the duality theorem. (hint: KKT condition)
'''
# The support vectors are circled in the previous figure. The duality theorem
# tells us that the support vectors are those whose Lagrange multipliers are
# non-zero. When the Lagrange multipler (aka. dual variable) is non-zero, the
# KKT conditions tell us that the constraint (from Problem 1) becomes 
# (w * x_i + b)y_i = 1. Thus, to find the support vectors, we can find the 
# points that satisfy this constraint.


# Question 2.5
'''
(1 point) Report the accuracy of your linear SVM classifier on both the 
training and test sets.
'''

training_accuracy = svm_classifier.score(X_train, y_train)
test_accuracy = svm_classifier.score(X_test, y_test)
print('Training Accuracy: ', training_accuracy)     # 1.0
print('Test Accuracy: ', test_accuracy)             # 1.0


# Question 2.6
'''
(1 point) What is the value of the margin? Justify your answer.
'''
# Recall that the margin is equal to 2/(||w||^2), where w is the weight vector.
weight = svm_classifier.coef_[0] # = [ 3.33266363 -3.33342658]
margin = 2 / (np.linalg.norm(weight)) # = 0.42430075463962524
# Thus, the margin is equal to approximately 0.424.


# Question 2.7
'''
(1 point) Which vector is orthogonal to the decision boundary?
'''
# As the decision boundary is defined by the equation w^T * x + b = 0, it follows 
# that the vector orthogonal to the decision boundary is the weight vector, w.



# Question 2.8
'''
(3 points) Split the iris dataset again in a training and test set, this time 
setting test size to 0.4 when calling train test split. Train the SVM classifier 
again. Does the decision boundary change? How about the test accuracy? Please 
justify why (hint: think about the support vectors), and illustrate your 
argument with a new plot.
'''


X_train, X_test, y_train, y_test = train_test_split(iris_hundred[:, :2], iris.target[:100], test_size=0.4, random_state=0)

# Create a linear SVM classifier
svm_classifier = SVC(kernel='linear', C=1000) 
svm_classifier.fit(X_train, y_train)

# Plot the decision boundary and margins
plt.figure(figsize=(8, 6))

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k', marker='o', s=100, label='Training Points')

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])

# Plot decision boundary and margins
Z = Z.reshape(xx.shape)
contour = plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'], label='_nolegend_')

# Plot support vectors
support_vectors = plt.scatter(svm_classifier.support_vectors_[:, 0], svm_classifier.support_vectors_[:, 1], s=200, facecolors='none',
            edgecolors='g', marker='o', linewidth = 1.5, label='Support Vectors')

legend_labels = ['Decision Boundary', 'Margins']

legend_handles = [
    Line2D([0], [0], color='k', linestyle='-', label=legend_labels[0]),
    Line2D([0], [0], color='k', linestyle='--', label=legend_labels[1]),
    support_vectors
]

plt.legend(handles=legend_handles)

plt.title('Linear SVM Classifier, test_size=0.4')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

# Accuracies
training_accuracy = svm_classifier.score(X_train, y_train)
test_accuracy = svm_classifier.score(X_test, y_test)
print('Training Accuracy: ', training_accuracy)     # 1.0
print('Test Accuracy: ', test_accuracy)             # 1.0



# print the weight vector and bias
w_second = svm_classifier.coef_[0]
b_second = svm_classifier.intercept_
print('Weight vector of SVM with test_size = 0.4: ', w_second) # [6.31804679 -5.26503723]
print('Bias of SVM with test_size = 0.4: ', b_second)          # [-17.32197688]


#  The test accuracy remains the same, 1.0 as does the training accuracy, 1.0, with both
# the training and test points falling outside of their respective margins.  

# Yes, the decision boundary changes (from the plot below, it is visible that the hyperplane
# corresponding to test_size = 0.4 is steeper). By decreasing test_size from 0.8 to 0.4, we
# increase the number of training points, and thus the number of support vectors
# (ie. finding points that are closer to the decision boundary). This results in a 
# tighter margin. 

 

# Question 2.9
'''
(1 point) Do the binary linear classifier and SVM have the same decision boundaries? 
The comparison should be made for the SVM obtained with test size=0.8.
'''

# Plot of the three decision boundaries
# Binary classifier
a_bin = -w_bin[0] / w_bin[1]
xx_boundary_bin = np.linspace(3.9, 7)
yy_boundary_bin = a_bin * xx_boundary_bin - (b_bin[0]) / w_bin[1]
plt.plot(xx_boundary_bin, yy_boundary_bin, 'k-', c='green', linewidth=2.5, label='Decision Boundary, Binary Classifier')
# SVM with test_size=0.8
a_first = -w_first[0] / w_first[1]
xx_boundary_first = np.linspace(3.9, 7)
yy_boundary_first = a_first * xx_boundary_first - (b_first[0]) / w_first[1]
plt.plot(xx_boundary_first, yy_boundary_first, 'k-', c='violet', linewidth=2.5, label='Decision Boundary, test_size=0.8')
# SVM with test_size=0.4
a_second = -w_second[0] / w_second[1]
xx_boundary_second = np.linspace(3.9, 7)
yy_boundary_second = a_second * xx_boundary_second - (b_second[0]) / w_second[1]
plt.plot(xx_boundary_second, yy_boundary_second, 'k-', c='orange', linewidth=2.5, label='Decision Boundary, test_size=0.4')
plt.legend(loc='upper right')
plt.show()

# No, they have different decision boundaries. According to the plot, the 
# binary classifier's decision boundary is slightly steeper than the SVM with 
# test_size = 0.8's decision boundary.  
# Also, we can see that their weight vectors and bias values are different:
# Binary classifier:        w = [ 1.80226162 -1.24492959], b = [-5.9685275]
# SVM with test_size = 0.8: w = [ 3.33266363 -3.33342658], b = [-7.66277845]


# Question 2.10
'''
(3 points) Now consider all 150 entries in the iris dataset, and retrain the SVM. 
You should find that the data points are not linearly separable. How can you deal 
with it? Justify your answer and plot the decision boundary of your new proposed 
classifier. For this question, use the SVM obtained with test size=0.4.
'''

iris_full = iris.data
X = iris_full[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_full[:, :2], iris.target, test_size=0.4, random_state=0)


# Polynomial SVM classifier
svm_classifier_poly = SVC(kernel='poly', C=1, random_state=0)
svm_classifier_poly.fit(X_train, y_train)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k', marker='o', s=100, label='Training Points')

# Meshgrid for the plot
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1.2, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the labels for each point in the meshgrid
Z = svm_classifier_poly.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)

plt.title('Polynomial SVM Classifier')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()


# By adding the 3rd class, the data points are no longer linearly separable. Thus,
# we can use a polynomial kernel to separate the data points.

training_accuracy = svm_classifier_poly.score(X_train, y_train)
test_accuracy = svm_classifier_poly.score(X_test, y_test)
print('Training Accuracy: ', training_accuracy)     # = 0.8444444444444444
print('Test Accuracy: ', test_accuracy)             # = 0.7

# Although the accuracy of the polynomial SVM classifier is not as high as the
# linear SVM classifier with only 2 classes, it is able classify the data points 
# with a reasonable accuracy. 