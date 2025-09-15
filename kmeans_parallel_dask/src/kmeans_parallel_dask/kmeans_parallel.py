#Libraries used
import numpy as np
import dask.array as da
from dask_ml.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.cluster import KMeans as SklearnKMeans

def phi(X,C):
    """
    cost function for a given set of centroids
    
    Input:
    - dataset X
    - set of centroids C
    Output:
    - Cost function
    """
    return pairwise_distances(X, C, metric='sqeuclidean').min(1).sum()

def assign_label(X,C):
    """
    Assign labels to each point of X corresponding to
    its closest centroid
    
    Input:
    - dataset X
    - set of centroids C
    Output:
    - set of labels label
    """
    return pairwise_distances_argmin_min(X, C, metric='sqeuclidean')[0]

def update_centroids(X,lab,k):
    """
    Function to update the position of the centroids
    
    Inputs:
    - dataset X
    - labels of the clusters lab
    - number of clusters k
    Output:
    - New centroids C
    """

    return da.stack([X[(da.asarray(lab) == i)].mean(0) for i in range(k)])


def kmeans__init(X: da.Array, k: int, l: int, random_state=42):
    """
    k-means|| initialization (Dask)

    Parameters
    ----------
    X : dask.array, shape (n_samples, n_features)
    k : int
    l : int, oversampling factor

    Returns
    -------
    C : np.ndarray, shape (k, n_features)
        Initial centroids
    """
    
    # 0) # RandomState for reproducibility
    ss = np.random.SeedSequence(random_state)
    child_seeds = ss.spawn(2)
    rng = np.random.default_rng(child_seeds[0])        # for Numpy
    rs  = da.random.RandomState(child_seeds[1].entropy) # for Dask

    n = X.shape[0]
    # 1) pick first center
    i0 = rng.integers(n)
    C = X[i0].compute()[None, :]

    # 2) initial cost
    psi = phi(X, C).compute()
    if psi <= 0:
        return C if len(C) == k else np.repeat(C, k, axis=0)[:k]
    
    
    
    # 3) O(log ψ) rounds
    t = max(1, int(np.ceil(np.log(psi))))
    for _ in range(t):
        d2min = pairwise_distances(X, C, metric="sqeuclidean").min(1)
        p = da.clip(l * d2min / psi, 0.0, 1.0)
        # campionamento bernoulliano con RandomState
        samples = rs.random_sample(size=p.shape, chunks=p.chunks) < p
        # indici dei True
        mask = da.where(samples)[0].compute()
        
        if mask.size:
            C = np.vstack([C, X[mask].compute()])
        psi = phi(X, C).compute()
        if psi <= 0:
            break

    # weights for k-means|| step 7
    dist2 = pairwise_distances(X, C, metric="sqeuclidean")   # dask-ml
    labels = da.argmin(dist2, axis=1)
    weights = da.bincount(labels, minlength=len(C)).compute().astype(float)

    # 8) recluster down to k
    k = min(k, C.shape[0])
    km = SklearnKMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(C, sample_weight=weights)
    return km.cluster_centers_


    

def kmeans_parallel(X, k, max_it=100, tol=1e-8, l=2):
    """
    Core function of our implementation: it implements kmeans with dask and kmeans|| initialization
    """
    C = kmeans__init(X, k, l) #initial centroids with k_means||
    for i in range(max_it): #iterative phase
        lab = assign_label(X, C)
        C_new = update_centroids(X, lab, k).compute()

        if da.allclose(C, C_new, atol=tol):
            #print(f"Main KMeans Converged after {i+1} iterations.")
            break

        C = C_new

    return lab, C