import math

import numpy
import numpy as np


# a is 'vector [3 4], b is 'vector' [1 2]
def euclidean_distance(data, centroids):
    # Euclidean distance (l2 norm)
    # 1-d scenario: absolute value
    # return abs(a-b)

    asdf_square = np.square(data - centroids)
    asdf_sum = np.sum(asdf_square)
    asdf_root = np.sqrt(asdf_sum)
    return asdf_root


# Step 1
def closestCentroid(x, centroids):
    assignments = []
    for i in x:
        # distance between one data point and centroids
        distance=[]
        for j in centroids:
            distance.append(euclidean_distance(i, j))
            # assign each data point to the cluster with closest centroid
        stuff1 = np.argmin(distance)
        assignments.append(stuff1) # np.argmin() output 'index' of the smallest value
    return np.array(assignments)


# Step 2
def updateCentroid(x, clusters, K):
    new_centroids = []
    for c in range(K):
        # Update the cluster centroid with the average of all points in this cluster
        # cluster_mean = x[clusters == c].mean()
        asdf = x[clusters == c]
        if len(asdf) == 0:
            print(">" * 10 + " Blank LIST value detected")
            return -1

        # need to modify this to calculate 2 dimensional 'average' [x y] [1 2]
        asdf_x = asdf[ : , 0]
        asdf_y = asdf[ : , 1]
        x_avg = np.mean(asdf[ : , 0])
        y_avg = np.mean(asdf[ : , 1])

        cluster_mean = [x_avg, y_avg]

        new_centroids.append(cluster_mean)
    return new_centroids


# 2-d kmeans
def kmeans(x, K):
    # initialize the centroids of 2 clusters in the range of [0,20)
    # asdf = np.random.rand(K)

    asdf = np.random.random( (K, K) )
    centroids = 14 * asdf
    centroids_start = centroids
    print('Initialized centroids: {}'.format(centroids))
    for i in range(10):
        clusters = closestCentroid(x, centroids)
        centroids = updateCentroid(x, clusters, K)

        if centroids == -1:
            print(">" * 5 + " Making new Centroids values")
            asdf = np.random.random( (K, K) )
            centroids = 14 * asdf
            centroids_start = centroids

        print('Iteration: {}, Centroids: {}'.format(i, centroids))

    return centroids, centroids_start


# 2 groups / centroid
K = 2

# x = np.array([0,2,10,12])
x = np.array([[2, 4],
              [1.7, 2.8],
              [7, 8],
              [8.6, 8],
              [3.4, 1.5],
              [9, 11]])

centroids, centroids_start = kmeans(x, K)
cen_np_array = np.array(centroids)
print(f"\n" + "-" * 20)
# x2 = numpy.append(x, asdf, axis=0)

def plot1():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import style
    X = np.array([[2, 4],
                  [1.7, 2.8],
                  [7, 8],
                  [8.6, 8],
                  [3.4, 1.5],
                  [9,11]])

    center1_x = centroids_start[ : , 0] # center1 BEFORE Kmeans()
    center2_y = centroids_start[ : , 1] # center2 BEFORE Kmeans()
    x_values = X[ : , 0] # everything at index 0
    y_values = X[ : , 1] # everything at index 1
    x2 = cen_np_array[:, 0] # center1 AFTER Kmeans()
    y2 = cen_np_array[:, 1] # center2 AFTER Kmeans()

    # need to graph the 'center' BEFORE the Kmeans() ran
    plt.scatter(center1_x, center2_y, s=200, marker="x") # 'graphing' the 'centers' before Kmeans algorithm ran
    plt.scatter(x_values, y_values, s=150) # 's' is the size of the dot
    plt.scatter(x2, y2, s=200, marker="s") # 'graphing' the 'centers' after Kmeans algorithm ran
    plt.show()


plot1()
print(f"debug wait")