#Libraries used
import dask
import dask.array as da
from dask.distributed import Client
from dask_ml.preprocessing import StandardScaler
from dask_ml.cluster import KMeans
import numpy as np
from dask_ml.metrics import pairwise_distances
from dask_ml.metrics import pairwise_distances_argmin_min
from time import time
from timeit import default_timer as timer
from dask_ml.datasets import make_blobs
import pandas as pd
from dask.distributed import SSHCluster
import matplotlib.pyplot as plt
import pickle
import numba


#K-means|| Initialization
def phi(X,C):
    return pairwise_distances(X, C, metric='sqeuclidean').min(1).sum() 
    # Compute the sum of distances to the nearest centroid at axis=1
    
def kmeans_parallel_init_dask(X, k, l):
    # Step 1: Randomly select a point from X
    n, d = X.shape
    idx = np.random.choice(n, size=1)
    C = X[idx].compute()  # Collect to memory for use

    # Step 2: Compute φX(C)
    phi_X_C = phi(X, C).compute() # Compute the sum of distances to the nearest centroid

    # Steps 3-6: Repeat O(log φX(C)) times
    rounds = int(np.log(phi_X_C))
    #print(f"Begin centroid sampling with number of rounds: {rounds}")
    for _ in range(rounds):
        dist_sq = pairwise_distances(X, C, metric='sqeuclidean').min(1)
        dist_sum = dist_sq.sum()
        p_x = l * dist_sq / dist_sum
        samples = da.random.uniform(size=len(p_x), chunks=p_x.chunks) < p_x
        sampled_idx = da.where(samples)[0].compute()
        
        new_C = X[sorted(sampled_idx)].compute() #https://github.com/dask/dask-ml/issues/39
        C = np.vstack((C, new_C))

    # Step 7: Compute weights
    dist_to_C = pairwise_distances(X, C, metric='euclidean')
    closest_C = da.argmin(dist_to_C, axis=1)

    weights = np.empty(len(C))
    counts = da.bincount(closest_C, minlength=len(C)).compute()
    weights[:len(counts)] = counts
    
    # Normalize weights so that they sum up to the number of centroids
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        raise ValueError("Sum of weights is zero, cannot normalize.")
    
    weights_normalized = weights / weight_sum
    
    dask_C = da.from_array(C, chunks=(C.shape[0], C.shape[1])) # here we ensure that the re-clustering occurs on a single-thread

    # Step 8: Recluster the weighted points in C into k clusters
    #print("Begin centroid re-clustering")
    labels, centroids = lloyd_kmeans_plusplus(X=dask_C, weights=weights_normalized,k=k, max_iters=10, tol=1e-8)

    return centroids

#k-means++ initialization
def kmeans_plusplus_init(X, weights, k):
    '''
    K-means++ initialization to select k initial centroids from X as a numpy array, keeping C as a NumPy array and weighting by provided weights.
    '''
    n, d = X.shape
    # Step 1: Randomly select the first centroid
    idx = np.random.choice(n, size=1)
    C = X[idx].compute()

    for _ in range(1, k):
        # Step 2: Compute distances from each point to the nearest centroid
        # C is a NumPy array, X is a Dask array
        # Compute the distances from each point to the nearest centroid normalizing by weights
        distances = pairwise_distances(X, C, metric='sqeuclidean').min(1) * (weights)
        
        # Compute the probabilities for choosing each point
        probabilities = distances / distances.sum()
        
        # Sample a new point based on these probabilities
        new_idx = np.random.choice(n, size=1, p=probabilities)
        new_centroid = X[sorted(new_idx)].compute()
        
        # Add the new centroid to the list
        C = np.vstack((C, new_centroid))
    return C


#update function K-means++ (auxiliary function)
def update_centroids_weighted(X, labels, weights, k):
    """
    Update the centroids by computing the weighted mean of the points assigned to each cluster.
    """
    centroids = []
    
    for i in range(k):
        # Select the points that are assigned to cluster i
        cluster_points = X[labels == i]
        
        # Select the corresponding weights for these points
        cluster_weights = weights[labels == i]
        
        if len(cluster_points) == 0:
            # If no points are assigned to this cluster, avoid division by zero
            # Continue without updating that centroid
            continue

        # Compute the weighted mean using dask.array.average
        weighted_mean = da.average(cluster_points, axis=0, weights=cluster_weights)
        
        # Append the computed weighted mean to the centroids list
        centroids.append(weighted_mean)
    
    # Convert centroids list to Dask array
    return da.stack(centroids)

#update function K-means++
def lloyd_kmeans_plusplus(X, weights, k, max_iters=100, tol=1e-8):
    """
    Lloyd's algorithm for k-means clustering using Dask weighting the mean to update centroids.
    """
    centroids = kmeans_plusplus_init(X, weights, k)

    for i in range(max_iters):
        labels = assign_clusters(X, centroids).compute()
        new_centroids = update_centroids_weighted(X, labels, weights=weights, k=k).compute()
        
        if da.allclose(centroids, new_centroids, atol=tol).compute():
            #print(f"Centroid Lloyd Converged after {i+1} iterations.")
            break
        
        centroids = new_centroids

    return labels, centroids

#Update (auxiliary function)
def assign_clusters(X, centroids):
    """
    Assign each point to the nearest centroid using Dask to parallelize the computation.
    """
    return pairwise_distances_argmin_min(X, centroids, metric='sqeuclidean')[0]

#Update (auxiliary function)
def update_centroids(X, labels, k):
    """
    Update the centroids by computing the mean of the points assigned to each cluster with Dask.
    """
    centroids = da.stack([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids

#True update function
def kmeans_parallel(X, k, max_iters=100, tol=1e-8, l=2):
    centroids = kmeans_parallel_init_dask(X, k, l)
    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k).compute()

        if da.allclose(centroids, new_centroids, atol=tol):
            #print(f"Main KMeans Converged after {i+1} iterations.")
            break

        centroids = new_centroids

    return labels, centroids