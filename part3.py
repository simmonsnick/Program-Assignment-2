import time
import warnings
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle


"""
Part 3.	
Hierarchical Clustering: 
Recall from lecture that agglomerative hierarchical clustering is a greedy iterative scheme that creates clusters, i.e., distinct sets of indices of points, by gradually merging the sets based on some cluster dissimilarity (distance) measure. Since each iteration merges a set of indices there are at most n-1 mergers until the all the data points are merged into a single cluster (assuming n is the total points). This merging process of the sets of indices can be illustrated by a tree diagram called a dendrogram. Hence, agglomerative hierarchal clustering can be simply defined as a function that takes in a set of points and outputs the dendrogram.
"""

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def data_index_function():
    return None


def compute():
    answers = {}

    """
    A.	Load the provided dataset “hierachal_toy_data.mat” using the scipy.io.loadmat function.
    """
    #Load the dataset
    data = scipy.io.loadmat('C:\\Users\\nsimm\\Downloads\\hierarchical_toy_data.mat')
    
    # return value of scipy.io.loadmat()
    answers["3A: toy data"] = data

    """
    B.	Create a linkage matrix Z, and plot a dendrogram using the scipy.hierarchy.linkage and scipy.hierachy.dendrogram functions, with “single” linkage.
    """
    #'Single' linkage method to construct the linkage matrix
    Z = sch.linkage(data['X'], method='single')

    
    #Plot dendrogram
    plt.figure(figure=(10, 5))
    sch.dendrogram(Z)
    plt.title('Dendrogram Single linkage')
    plt.xlabel('Data points')
    plt.ylabel('Distance')
    plt.show()
    # Answer: NDArray
    answers["3B: linkage"] = Z

    # Answer: the return value of the dendogram function, dicitonary
    dendrogram_return_value = sch.dendrogram(Z)
    answers["3B: dendogram"] = dendrogram_return_value

    """
    C.	Consider the merger of the cluster corresponding to points with index sets {I={8,2,13}} J={1,9}}. At what iteration (starting from 0) were these clusters merged? That is, what row does the merger of A correspond to in the linkage matrix Z? The rows count from 0. 
    """
    Z = [ [8, 2, 0.5, 3],
          [1, 9, 0.7, 2], 
          [8, 9, 0.8, 5] ]
    
    #Index sets 
    I = {8, 2, 13}
    J = {1, 9}
    
    for idx, row in enumerate(Z):
        if (set(row[:2]) == I  or set(row[:2]) == J):
            print(f"Clusters that are merged at iteration")
    
    # Answer type: integer
    answers["3C: iteration"] = -1

    """
    D.	Write a function that takes the data and the two index sets {I,J} above, and returns the dissimilarity given by single link clustering using the Euclidian distance metric. The function should output the same value as the 3rd column of the row found in problem 2.C.
    """
def single_link_dissimilarity(data, I, J):
    #Extract points from data I and J
    I = data[np.array(list(I)) - 1]
    J = data[np.array(list(J)) - 1]
    
    #Claculate pairwise distances between points 
    distances = np.linalg.norm(I[:, np.newaxis, :] - J, axis=2) 
    
    #Minimum distance
    min_distance = np.min(distances)
    #return min_distance

    # Answer type: a function defined above
    answers["3D: function"] = data_index_function
    

    """
    E.	In the actual algorithm, deciding which clusters to merge should consider all of the available clusters at each iteration. List all the clusters as index sets, using a list of lists, 
    e.g., [{0,1,2},{3,4},{5},{6},…],  that were available when the two clusters in part 2.D were merged.
    """
    Z = [ [0, 1, 0.1, 2],
          [2, 3, 0.2, 3],
          [4, 5, 0.3, 4]
        ]
def get_clusters(Z):
    clusters = []
    num_points = len(Z) + 1
    for row in Z:
        idx_I, idx_J = int(row[0]), int(row[1])
        # Merge clusters to idx_I and idx_J
        new_cluster = set()
     
    all_clusters = get_clusters(Z)   
    I = {8, 2, 13}
    J = {1, 9}
    
    for idx, cluster in enumerate(all_clusters):
        if I in cluster and J in cluster:
            merge_iteration = idx
    
    clusters_at_merge = all_clusters[merge_iteration]
    print(clusters_at_merge)
    
    clusters = [
        {0, 1, 2},
        {3, 4},
        {5},
        {6}
    ]
    clusters_list = [{idx for idx, _ in enumerate(cluster)} for cluster in clusters]
    # List the clusters. the [{0,1,2}, {3,4}, {5}, {6}, ...] represents a list of lists.
    answers["3E: clusters"] = clusters_list

    """
    F.	Single linked clustering is often criticized as producing clusters where “the rich get richer”, that is, where one cluster is continuously merging with all available points. Does your dendrogram illustrate this phenomenon?
    """

    # Answer type: string. Insert your explanation as a string.
    answers["3F: rich get richer"] = "Yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
