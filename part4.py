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
from scipy.spatial.distance import pdist

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle


"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster():
    return None

def fit_modified():
    return None


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """
def fit_hierarchical(dataset, linkage_type='ward', n_clusters=2):
    data, labels = dataset
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    hierarchical_clustering.fit(scaled_data)
    
    datasets = ['nc', 'nm', 'bvv', 'add', 'b']

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)

    dct = answers["4A: datasets"] = {}
    
    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """
    
def make_plots(datasets, linkage_type, n_clusters):
    fig, axes = plt.subplots(len(datasets), len(linkage_type), figsize=(15,10))
    
    for i, (dataset_abbr, dataset) in enumerate(datasets.items()):
       data, true_labels = dataset
    for j, linkage_type in enumerate(linkage_types):
        predict_labels = fit_hierarchical_cluster(dataset, linkage_type=linkage_type, n_clusters=n_clusters)
        
        #Plot clusters
        ax = axes[i, j]
        ax.scatter(data[:, 0], data[:, 1], c=predict_labels, cmap='viridis', s=50, edgecolors='k')
        ax.set_title(f"{dataset_abbr.upper()} - {linkage_type()}")
        ax.set_xticks(())
        ax.set_yticks(())
        
    plt.tight_layout()
    plt.savefig("hierarchical_clustering_comparison.jpg")
    plt.show()
   
    dataset_abbr = ['nc', 'nm', 'bvv', 'add', 'b']
    datasets = {}
    
    linkage_types = ['single', 'complete', 'ward', 'average']
    
    n_clusters = 2
    
    make_plots(datasets, linkage_type, n_clusters)

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = [""]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """
def fit_modified(dataset):
    
    data, labels = dataset
    
    distances = pdist(data)
    
    Z = linkage(distances, method='ward')
    rate_of_change = np.diff(Z[:, 2])
    
    max_rate_index = np.argmax(rate_of_change)
    
    cut_off_distance = Z[max_rate_index, 2]
    return cut_off_distance
    
def make_plots(datasets, cut_off_distances):
    fig, axes = plt.subplots(len(datasets), figsize=(8, 6))
    
    for i, (dataset_abbr, dataset) in enumerate(datasets.items()):
        data, true_labels = dataset
        
        # Apply hierarchical clustering with cut-off distance
        hierarchical_clustering = AgglomerativeClustering(distance_threshold=cut_off_distances[i], n_clusters=None)
        predicted_labels = hierarchical_clustering.fit_predict(data)
        
        
        # Plot the clusters
        ax = axes[i]
        dendrogram(linkage(data, method='ward'), ax=ax, truncate_mode='level', color_threshold=cut_off_distances[i])
        ax.set_title(f"{dataset_abbr.upper()} - Cut-off Distance: {cut_off_distances[i]:.2f}")
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Distance')
        ax.axhline(y=cut_off_distances[i], color='r', linestyle='--')
    
        plt.tight_layout()
        plt.savefig("cut_off_distance_plots.jpg")
        plt.show()

    
    # dct is the function described above in 4.C
    dct = answers["4A: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()
    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
