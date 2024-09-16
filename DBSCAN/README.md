
```markdown
# Machine Learning for Engineering Applications

## Overview

This repository contains implementations and resources for a machine learning module focused on engineering applications. The project covers data exploration, clustering, and normalization using Python. Key libraries include NumPy, scikit-learn, and Matplotlib.

## Project Description

The project involves clustering used car sales data with the DBSCAN algorithm. The main components are:

1. **Loading and Normalizing Data**: Data is loaded from a `.npy` file, normalized, and then clustered.
2. **Clustering with DBSCAN**: Apply the DBSCAN algorithm with default parameters and visualize the results.
3. **Hyperparameter Tuning**: Optimize the `epsilon` parameter for DBSCAN through grid search and evaluate clustering performance using silhouette scores.
4. **Visualization of Optimal Clustering**: Visualize the clustering results using the optimal `epsilon` value.

Additionally, a custom Z-score normalization class is provided for data preprocessing.

## Files and Scripts

### 1. Data Loading and Clustering

- **File**: `clustering.py`
- **Description**: This script performs the following:
  - Loads the dataset from `secondary_hand_car_sales.npy`.
  - Extracts relevant features: year, mileage, and price.
  - Normalizes the data using Z-score normalization.
  - Applies DBSCAN clustering with both default and optimized parameters.
  - Visualizes the clustering results in terms of year vs mileage and year vs price.

### 2. Hyperparameter Tuning

- **File**: `hyperparameter_tuning.py`
- **Description**: This script performs:
  - Grid search for the optimal `epsilon` parameter for DBSCAN.
  - Evaluation of clustering performance using silhouette scores.
  - Visualization of the number of clusters and silhouette coefficients as functions of `epsilon`.

### 3. Z-Score Normalization

- **File**: `zscorer.py`
- **Description**: Contains a custom class for Z-score normalization, which scales data by:
  - Computing the mean and standard deviation.
  - Transforming data using Z-score normalization.
  - Inverse-transforming data to the original scale.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Machine-Learning-for-Engineering-Applications.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Machine-Learning-for-Engineering-Applications
   ```
3. Install the required packages:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

## Usage

1. Run clustering with default parameters:
   ```bash
   python clustering.py
   ```
2. Perform hyperparameter tuning:
   ```bash
   python hyperparameter_tuning.py
   ```
3. Test Z-score normalization:
   ```bash
   python zscorer.py
   ```


## Acknowledgements

- NumPy
- Matplotlib
- scikit-learn
