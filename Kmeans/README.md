# K-Means Clustering Algorithm

This repository implements a custom K-Means clustering algorithm from scratch using NumPy, as well as a comparison with the `sklearn` K-Means implementation. It includes core functions such as centroid initialization, cluster assignment, centroid updates, and convergence checks. Additionally, it features plotting capabilities to visualize the clustering results.

## Features

- **Custom K-Means Implementation:**
  - Cluster assignment using L1 or L2 norms.
  - Centroid updates using mean (L2) or median (L1).
  - Detects and relocates empty clusters.
  - Convergence criterion based on cluster assignment changes.
  - Handles clusters of arbitrary dimensionality.
  
- **Visualization:**
  - Plots clusters with their centroids.
  - Comparisons between custom and `sklearn` K-Means results.

## Files

- `main.py`: The main script containing the K-Means clustering implementation and an example usage with a test dataset.
- `utils_clustering.py`: Utility functions such as centroid initialization, plotting, and auxiliary methods like `sse`, `bss`, and `find_farthest_point`.
- `example_data_Kmeans.csv`: A dataset for testing and visualizing the K-Means algorithm.
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/kmeans_clustering.git
   cd kmeans_clustering
   ```

2. Install required dependencies:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

## Usage

To run the K-Means clustering algorithm with a sample dataset, use the following steps:

1. **Run the main script**:
   ```bash
   python main.py
   ```

   This will:
   - Load a test dataset `x` and apply both the custom K-Means implementation and the `sklearn` K-Means implementation.
   - Plot the clustering results.
   - Print out the updated labels and centroid positions after clustering.

2. **Customize Parameters**:
   You can modify the `kmeans_clustering()` function in the `main.py` script to experiment with different norms (`L1`, `L2`), initial centroids, or datasets.

### Example Workflow:

```python
from main import kmeans_clustering

# Define data points
x = np.array([[1, 9], [2, 7], [3, 8], [4, 3], [5, 2], [7, 2], [8, 4], [6, 4], [10, 11], [10, 9], [8, 11], [12, 9]])

# Run the K-Means clustering
labels, centroids, cost = kmeans_clustering(x=x, K=3, norm='L2')

# Plot the results
plot_clusters(x=x, labels=labels, centroids=centroids)
```

## Function Descriptions

### Core Functions

- **`assign_cluster()`**: 
  - Assigns data points to the nearest cluster centroid based on the specified norm (L1 or L2).
  
- **`update_centroids()`**: 
  - Recomputes centroids based on the current cluster assignments, using either the mean (L2) or median (L1).
  
- **`is_converged()`**: 
  - Checks if the K-Means algorithm has converged by monitoring changes in cluster assignments over iterations.

- **`kmeans_clustering()`**: 
  - The main K-Means loop that updates cluster assignments and centroids until convergence or a maximum iteration limit is reached.

- **`relocate_empty_centroid()`**: 
  - Handles the case where a cluster becomes empty by relocating the centroid far away from existing ones.

### Utility Functions (from `utils_clustering.py`)

- **`pick_random_points()`**: Randomly selects `K` points from the dataset to initialize centroids.
- **`sse()`**: Computes the sum of squared errors (inertia) for cluster evaluation.
- **`bss()`**: Computes the between-cluster sum of squares.
- **`find_farthest_point()`**: Finds the point in the dataset farthest from existing centroids.
- **`plot_clusters()`**: Plots clusters with their centroids.

## Testing with Sklearn

The repository also includes a comparison with the `sklearn` implementation of K-Means:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
labels_sklearn = kmeans.labels_
centroids_sklearn = kmeans.cluster_centers_

plot_clusters(x=data, centroids=centroids_sklearn, labels=labels_sklearn)
```

## Visualizing Results

The code provides several visualizations to help understand the clustering process, including:
- Raw data without clustering
- Clustered data points with their respective centroids
- A comparison of custom K-Means vs. `sklearn` K-Means
