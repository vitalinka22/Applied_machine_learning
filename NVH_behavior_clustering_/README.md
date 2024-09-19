# NVH Behavior Clustering

This repository contains a script for analyzing and clustering brake noise and vibration (NVH) data using the DBSCAN clustering algorithm. The `NVH_behavior_clustering.py` script performs data cleaning, normalization, and clustering of brake noise data based on frequency and sound pressure level (SPL).

## Overview

The script processes brake data to identify and cluster noisy brake stops. It performs the following steps:

1. **Data Loading and Preprocessing**: Load brake data, clean it, and remove outliers.
2. **Data Analysis**: Calculate statistics, visualize distributions, and identify noisy data.
3. **Data Normalization and Clustering**: Normalize data and apply DBSCAN clustering.
4. **Visualization**: Plot cleaned and clustered data for analysis.

## Dependencies

- `numpy`
- `matplotlib`
- `sklearn`

To install the required packages, use the following command:

```
pip install numpy matplotlib scikit-learn
```

## Script Description

### Data Loading and Preprocessing

- **Load Data**: The script reads data from `brake_data_assignment.txt`.
- **Feature Extraction**: Extracts and prints the range, mean, and median for each feature channel.
- **Data Cleaning**: Removes data with negative brake pressure, unrealistic velocities, excessive steering angles, non-audible noise, and non-audible sound levels.

### Data Analysis

- **Statistics**: Computes and prints statistics for each feature before and after cleaning.
- **Histograms**: Visualizes the distribution of feature values before and after cleaning.
- **Scatter Plots**: Identifies and plots noisy data points versus non-noisy ones.

### Data Normalization and Clustering

- **Normalization**: Normalizes frequency and SPL values to a common range.
- **Clustering**: Applies DBSCAN clustering to the normalized data.
- **Visualization**: Displays the clustering results in a plot.

### Visualization

- **Feature Distributions**: Plots histograms of feature distributions before and after cleaning.
- **Cluster Visualization**: Shows cleaned data and clusters in a frequency vs SPL plot.

## Usage

Ensure that `brake_data_assignment.txt` is in the same directory as the script. To run the script, use:

```
python NVH_behavior_clustering.py
```

## Example Output

- **Histograms**: Distribution of feature values before and after cleaning.
- **Scatter Plots**: Visualization of noisy vs non-noisy data.
- **Cluster Visualization**: Plot of frequency vs SPL showing clustering results.

## Notes

- **DBSCAN Parameters**: Adjust `eps` and `min_samples` in the DBSCAN algorithm based on your data and scaling method.
- **Data Format**: Ensure the input data is correctly formatted and available as `brake_data_assignment.txt`.

For questions or issues, please open an issue on the [GitHub repository](https://github.com/vitalinka22/Applied_machine_learning).
