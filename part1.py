import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def load_circle_data():
    data, labels = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=42)
    return data, labels
def load_moons_data():
    data, labels = make_moons(n_samples=100, noise=0.05, random_state=42)
    return data, labels
def load_blobs_data():
    data, labels = make_blobs(n_samples=100, random_state=42)
    return data, labels
def load_transformed_data():
    X, y = make_blobs(n_samples=100, random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    data = np.dot(X, transformation)
    return data, y

def fit_kmeans():
    return None

def compute():
    answers = {}
    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """
 #Check random state
    random_state = 42
    n_samples = 100
    
    #5 datasets
    datasets = ['nc', 'nm', 'bvv', 'add', 'b']
    dct = {}
    
    #Load Dataset
    for dataset in datasets:
        if dataset == 'nc':
            data, labels = load_circle_data()
        elif dataset == 'nm':
            data, labels = load_moons_data()
        elif dataset == 'bvv':
            data, labels = load_blobs_data()
        elif dataset == 'add':
            data, labels = load_transformed_data()
        elif dataset == 'b':
            data, labels = load_blobs_data()
        
        #Store data in dct
        dct[dataset] = [data, labels]
    
    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    answers["1A: datasets"] = dct
    #print(answers)
    
    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """
def fit_kmeans(data, n_clusters):     #fit_means clustering being provide in this dataset 
        
   #Extract this dataset and labels ERROR
    X, y = data
        
    #Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #KMeans for the clustering
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(X)
    
    #predict the labels here
    predict_labels = kmeans.labels_
    
    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans
    dct = fit_kmeans
    
    #return predict_labels

    """"
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """
def plot_part1C(datasets_dict):
        
        #Datasets store 
        #datasets = ['nc', 'nm', 'bvv', 'add', 'b']
        
        #Figure with 4 rows and 5 columns
        fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 16))
        k_values = [2, 3, 5, 10]
        
        #Each dataset to column 
        for i, k in enumerate(k_values):
            for j, (dataset_name, dataset) in enumerate(datasets_dict.items()):
                predict_labels = fit_kmeans(dataset, k)
                if i == 0:
                    axes[i, j].set_title(dataset)
                if j == 0:
                    axes[i, j].set_ylabel(f'k={k}')
                
                #Scatter plot with labels and colors
                axes[i, j].scatter(dataset[:, 0], dataset[:, 1], c=predict_labels, cmap='viridis', s=10)
                axes[i, j].set_title(f'Dataset {i+1}, k={k}')
                axes[i, j].set_xlabel("x")
                axes[i, j].set_ylabel("y")
                
            plt.tight_layout()
            plt.show
            #plt.savefig()
            
    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
        dct = answers["1C: cluster successes"] = {"bvv": [3], "b": [3]} 

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
        dct = answers["1C: cluster failures"] = ["nc", "nm"]

        """
D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """
        datasets = ['nc', 'nm', 'bw', 'add', 'b']
        k_values = [2, 3]
        
def fit_kmeans(data, n_clusters):
    
    X , y = data
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(X)
    return kmeans.labels_
    
def sensitivity_data(datasets, k_values):
    
    sensitivity = []
    for data, X in enumerate(datasets):
        for k in k_values:
            labels_initial = fit_kmeans((X), k)
            sensitivity_dataset = []
            
            for y in range(5):
                labels_new = fit_kmeans((X), k)
                sensitivity_dataset.append(np.array_equal(labels_initial, labels_new))
                
            sensitivity.append((f'Dataset {data+1}, k={k}'))
            
    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
            dct = answers["1D: datasets sensitive to initialization"] = [""]
            return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
