import dask.array as da
from kmeans_parallel_dask import kmeans_parallel

def test_runs():
    X = da.random.random((1000, 3), chunks=(200, 3))
    labels, centroids = kmeans_parallel(X, k=3, l=2, max_iters=5)
    assert centroids.shape == (3, 3)
