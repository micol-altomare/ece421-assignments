# ECE421 - Introduction to Machine Learning
# Assignment 1

from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

# Problem 1

# 1.1
'''
(5 points) Implement κ-means yourself. Your function should take in an array containing
a dataset and a value of κ, and return the cluster centroids along with the cluster
assignment for each data point. You may choose the initialization heuristic of your
choice among the two we saw in class. Hand-in the code for full credit. For this
question, you should not rely on any library other than NumPy in Python.
'''

# Load the breast cancer dataset
dataset = load_breast_cancer()

def k_means(dataset, k):
    '''Implement the k-means clustering algorithm. The heuristic is choosing k random 
    initial centroids from the dataset.
    '''
    centroids = []
    prev_centroids = np.zeros((k, len(dataset.data[0])))
    
    # Random initialization of k centroids
    for i in range(k): 
        random_index = np.random.randint(0, len(dataset.data) - 1)
        centroid_chosen = dataset.data[random_index] 
        centroids.append(centroid_chosen) #***could clean up 

    max_iterations = 10000
    w = 0
    
    for w in range(max_iterations):
        # while not np.allclose(centroids, prev_centroids): 

        assignments = [[] for i in range(k)]

        print("Iteration: ", w)

        # Cluster assignment
        # compute the distance between each data point and each centroid
        for point in dataset.data: 
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            centroid_index = np.argmin(distances) # the index of the centroid with the minimum distance
            assignments[centroid_index].append(point) # the point is assigned by adding it to the list of points whose centroid is the index in assignments

        prev_centroids = np.copy(centroids)

        # Move centroids
        for r in range(len(centroids)):    
            # compute the mean of the assigned data points
            mean = np.mean(np.array(assignments[r]), dtype=np.float64, axis=(0)) # column-wise # TODO: double-check
            # move the centroid to the mean
            centroids[r] = mean
        
        if np.allclose(centroids, prev_centroids, rtol=1e-09): # convergence check # TODO: use divergence instead?
            print("Converged!")
            break

        w += 1
        
    return centroids, assignments


# 1.2
'''
(1 point) Run the κ-means algorithm for values of κ varying between 2 and 7, at increments 
of 1. Justify in your answer which data you passed as the input to the κ-means algorithm.
'''

for i in range(2, 8):
    k_means(dataset, i)

# I chose to pass in the entire dataset as the input to the k-means algorithm since
# the dataset isn't computationally expensive and all the points can be used, via 
# distortion to help determine the best value of k.


# 1.3
'''
(2 points) Plot the distortion achieved by κ-means for values of κ varying between 2 and 7, 
at increments of 1. Hand-in the code and figure output for full credit. For this question, 
you may rely on plotting libraries such as matplotlib.
'''

def distortion(centroids, assignments):
    '''
    Compute the distortion (ie. minimize the objective) of the k-means clustering algorithm 
    given the final (post-convergence) centroids and assignments.
    '''
    distances = [] # a list where each element is the distance between a point in the dataset and its centroid
    total_distance = 0
    m = 0
    
    print(len(assignments))
    print(len(assignments[0]))

    for i in range(len(centroids)): # iterate per centroid
        # add the sum of the distances between each point (found in assignments[i]) and its centroid (centroids[i])
        distances += [np.linalg.norm(its_point - centroids[i])**2 for its_point in assignments[i]] # TODO: verify distortion equation
        m += len(assignments[i])

    total_distance = sum(distances)
    print("Total distance: ", total_distance)
    print("m: ", m)

    j = total_distance / m

    return j


# Run k-means and compute the distortion for each value of k
k_vals = np.arange(2, 8, 1)
j_vals = np.zeros(len(k_vals))

for i in range(len(k_vals)):
    print("=====================================")
    print("Running k-means for k = ", k_vals[i], "...")
    centroids, assignments = k_means(dataset, k_vals[i])
    j_vals[i] = distortion(centroids, assignments)

print("Done!")

# Plot the k vs. distortion graph
plt.plot(k_vals, j_vals)
plt.xlabel('k-value')
plt.ylabel('Distortion (J)')
plt.show()

print("Actually Done!")

# 1.4
'''
(1 point) If you had to pick one value of κ, which value would you pick? Justify your choice.
'''
# I'd choose k = 4 since it is the point on the graph at which the distortion begins to level 
# off (is minimized) and when real-world constraints would likely make it impractical to have 
# more clusters.
# TODO: verify my answer for k