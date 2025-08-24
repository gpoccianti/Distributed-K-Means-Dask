# kmeans_parallel_dask

Minimal Dask implementation of k-means|| initialization + Lloyd iterations.

## Usage
```python
import dask.array as da
from kmeans_parallel_dask import kmeans_parallel

X = da.random.random((10000, 2), chunks=(1000, 2))
labels, centroids = kmeans_parallel(X, k=5, l=2, max_iters=50)
