# K-means Clustering

## Overview

This project implements the K-means clustering algorithm using NumPy, along with utility functions for evaluating clustering performance and visualizing results.

## Files

- `kmeans/main.py`: Main script to run K-means clustering. Includes functions for clustering, centroid updates, and convergence checking.
- `kmeans/utils_clustering.py`: Utility functions for clustering evaluation (SSE, BSS), centroid initialization, and plotting.
- `kmeans/example_data_Kmeans.csv`: Example dataset for clustering experiments.

## Usage

1. **Run the Main Script**

   Execute the main script to test K-means clustering:

   ```bash
   python kmeans/main.py
   ```

   This will:
   - Load and plot example data.
   - Run K-means clustering on the data.
   - Compare results with scikit-learn's KMeans.
   - Display clustering results and evaluation metrics.

2. **Functions**

   - `assign_cluster(x, centroids, norm='L2')`: Assigns data points to clusters based on the nearest centroid.
   - `update_centroids(x, labels, K, norm='L2')`: Updates centroid positions.
   - `is_converged(centroids, labels)`: Checks if the algorithm has converged.
   - `kmeans_clustering(x, K, norm='L2', init_centroids=None)`: Performs K-means clustering.

3. **Utilities**

   - `sse(x, centroids, labels, norm='L2')`: Computes Sum of Squared Errors.
   - `bss(x, centroids, norm='L2')`: Computes Between-cluster Sum of Squares.
   - `plot_clusters(x, labels=None, centroids=None)`: Plots clusters and centroids.

## Requirements

- NumPy
- Matplotlib
- scikit-learn (for comparison)

