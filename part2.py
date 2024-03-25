from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #
from sklearn.cluster import KMeans

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans():
    return None

def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    data_blob = make_blobs(center_box=(-20, 20), n_samples=20, centers=5, random_state=12)

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = {}
    
    dct = answers["2A: blob"] = [np.zeros(0)]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """
def fit_kmeans(data, n_clusters):
    X, y = data
    
    #Scaling the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #KMeans for clustering
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(X)
    
    #Predict kmean labels
    predict_labels = kmeans.labels_
    
    #SSE calcuations 
    SSE = kmeans.inertia_
    
    # dct value: the `fit_kmeans` function
    answers = {}
    dct = answers["2B: fit_kmeans"] = fit_kmeans
    return predict_labels, SSE
    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
def compute_sse(data, k_values):
    SSE_values = []
    for k in k_values:
        SSE = fit_kmeans(data, k)
        SSE_values.append(SSE)
        return SSE_values
    
    #k values for k=1,2, ...,8
    k_values = range(1, 9)
    
    #Compute SSE for each k value
    SSE_values = compute_sse(data, k_values)
    
    #Plot SSE as a function of k
    plt.plot(k_values, SSE_values, marker='o')
    plt.xlabel('Clusters k')
    plt.ylabel('SSE')
    plt.title('Optimal k')
    plt.xticks(k_values)
    plt.grid(True)
    plt.ion()
    plt.show()
    
    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    sse_plot_data = [[0.0, 100.], [1, 200.]]
    
    answers = {}
    
    dct = answers["2C: SSE plot"] = sse_plot_data
    return answers
    
    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k's agree?
    """
def compute_inertia(data, k_values):
    inertia_values = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='random', random_state=42)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
        return inertia_values
    
    plt.plot(k_values, inertia_values, marker='o')
    plt.xlabel('Clusters k')
    plt.xlabel('Inertia')
    plt.title('Optimal k on Inertia')
    plt.grid(True)
    plt.show()
    
    optimal_k_inertia = np.argmin(inertia_values) + 1
    
    answers = {}
    
    inertia_plot_data = [[0.0, 100.], [1, 200]]
    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = inertia_plot_data

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers

# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
