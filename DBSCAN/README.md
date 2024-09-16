# DBSCAN and Z-Score Normalization Projects

## Overview

This repository contains two main components for data analysis:

1. **Clustering with DBSCAN**: Applies the DBSCAN clustering algorithm to second-hand car sales data, optimizes parameters, and visualizes the results.
2. **Z-Score Normalization**: Implements a custom Z-score normalization for data preprocessing.

## Project Descriptions

### 1. DBSCAN Clustering

- **Path**: `DBSCAN/second_hand_car_sales.py`
- **Description**:
  - Loads and processes second-hand car sales data from a `.npy` file.
  - Performs DBSCAN clustering with default parameters and visualizes the results.
  - Optimizes the `epsilon` parameter through grid search and evaluates clustering performance using silhouette scores.
  - Generates and saves plots for clustering with default and optimal parameters, and a hyperparameter study.

### 2. Z-Score Normalization

- **Path**: `DBSCAN/zsore.py`
- **Description**:
  - Implements a custom class for Z-score normalization.
  - Reads data from a CSV file, applies Z-score transformation, and prints the original, transformed, and re-transformed data.
  - Assesses the accuracy of the transformation by comparing the original and re-transformed data.

## Files

- `DBSCAN/second_hand_car_sales.py`: Script for clustering second-hand car sales data using DBSCAN.
- `DBSCAN/secondary_hand_car_sales.npy`: Data file used for clustering.
- `DBSCAN/zsore.py`: Script for Z-score normalization.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vitalinka22/Applied_machine_learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Applied_machine_learning
   ```
3. Install the required packages:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

## Usage

1. To run DBSCAN clustering:
   ```bash
   python DBSCAN/second_hand_car_sales.py
   ```
2. To run Z-score normalization:
   ```bash
   python DBSCAN/zsore.py
   ```

## Acknowledgements

- NumPy
- Matplotlib
- scikit-learn

