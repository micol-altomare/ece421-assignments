# ECE421 - Introduction to Machine Learning
# Assignment 1

from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib as plt

# Problem 1

# 1.1
'''
(5 points) Implement κ-means yourself. Your function should take in an array containing
a dataset and a value of κ, and return the cluster centroids along with the cluster
assignment for each data point. You may choose the initialization heuristic of your
choice among the two we saw in class. Hand-in the code for full credit. For this
question, you should not rely on any library other than NumPy in Python.
'''

# Load the breast cancer dataset (already is a numpy.ndarray)
dataset = load_breast_cancer()
data = dataset.data # data is a numpy array
# or just load_breast_cancer().data

def pick_k(dataset, num_clusters):
    '''pick k'''
    #for i in range(1, num_clusters):
    #    # pick k random centroids from the dataset and name the aray clusters_i, where i is the cluster number based on the for loop
    #    f'centroids_{i}' = np.random.choice(dataset, num_clusters, replace=False) # replace=False means no duplicates
    global initial_centroids
    initial_centroids = {}

    for i in range(1, num_clusters + 1):
        centroid_index = np.random.randint(0, len(dataset.data) - 1)
        initial_centroids['centroids_' + str(i)] = dataset.data[centroid_index] # replace=False means no duplicates (to avoid duplicate centroids)
        # remove the centroid from the dataset so that it is not chosen again
        dataset.data = np.delete(dataset.data, centroid_index, axis=0)
        
        # run k-means, compute the distortion, pick centroids with the lowest distortion

    return None # TODO: return k

def k_means(dataset, k): # let k = 3
    '''Implement the k-means clustering algorithm. The heuristic is choosing k random 
    initial centroids from the dataset.
    '''
    centroids = []
    assignments = []
    

    # Random initialization of k centroids
    for i in range(k): #***checked and this loop works!
        #print(dataset.data.shape)
        #print(len(centroids)))
        centroid_index = np.random.randint(0, len(dataset.data) - 1)
        centroid_chosen = dataset.data[centroid_index] # initialize the centroids
        print(centroid_chosen)
        # add k_centroids to the centroids list
        centroids.append(centroid_chosen)
        print(centroids)
        print(len(centroids))
        # remove the centroid from the dataset so that it is not chosen again
        dataset.data = np.delete(dataset.data, centroid_index, axis=0)
        #print(dataset.data.shape)
        #print(len(centroids))



    # Cluster assignment

    for i in range(len(dataset.data)): #****checked and it works as expected!
        prev_distance = 100000000
        the_nearest_centroid = None

        # compute the distance between each data point and each centroid
        for centroid in centroids:
            # compute the distance between dataset.data[i] and each centroid
            distance = np.linalg.norm(dataset.data[i] - centroid)
            print("Distance:" + str(distance))
            print("Previous distance:" + str(prev_distance))

            if distance < prev_distance:
                the_nearest_centroid = centroid
                print("New centroid and the distance was updated to the be the new distance.")
                print("The nearest centroid:" + str(the_nearest_centroid))
                print("The point:"+ str(dataset.data[i])) #***********
                prev_distance = distance
                print("New previous distance:" + str(prev_distance))

        # assign the data point to the nearest centroid
        assignments.append([dataset.data[i], the_nearest_centroid])
        print("The assignment:" + str(assignments[i])) #***********
        print("The nearest centroid:" + str(the_nearest_centroid))


    # Move centroids
    for centroid in centroids:
        print("Current centroid:" + str(centroid))
        # iterate through the dictionary and add any points whose centroid is centroid to a new list
        corresponding_points = []
        for i in range(len(assignments)):
            print("The assignment: " + str(assignments[i]))
            print("The centroid: " + str(assignments[i][1]))
            print(type(assignments[i][1]))
            print(type(centroid))
            if np.array_equal(assignments[i][1], centroid):
                corresponding_points.append(assignments[i][0])
                print("Appended!")

        print("The corresponding points:" + str(corresponding_points))
        # compute the mean of the assigned data points
        corresponding_points = np.array(corresponding_points)
        print(type(corresponding_points))
        print(len(corresponding_points))
        # print the dimension of the array
        print(corresponding_points.shape)
        # compute the mean of the corresponding points along each axis
        mean = np.mean(corresponding_points, dtype=np.ndarray, axis=(1)) # column-wise
        print("Mean:"+ str(mean))
        print(len(mean)) #***********
        print(type(mean)) #***********
        # move the centroid to the mean
        centroid = mean
        print(centroid)

    return centroids, assignments


# 1.2
'''
(1 point) Run the κ-means algorithm for values of κ varying between 2 and 7, at increments 
of 1. Justify in your answer which data you passed as the input to the κ-means algorithm.
'''

k_means(dataset, 3)
for i in range(2, 7, 1):
    print(k_means(dataset, i))


# 1.3
'''
(2 points) Plot the distortion achieved by κ-means for values of κ varying between 2 and 7, 
at increments of 1. Hand-in the code and figure output for full credit. For this question, 
you may rely on plotting libraries such as matplotlib.
'''

# 1.4
'''
(1 point) If you had to pick one value of κ, which value would you pick? Justify your choice.
'''

#if __name__ == "__main__":