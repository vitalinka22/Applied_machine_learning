# Decision Tree Preparations

This script implements essential functionalities for building decision trees, including calculating entropy, information gain, and finding the best splits. It also provides visualization for decision boundaries based on the dataset.

## Overview

The `decision_tree_preparations.py` script includes functions for:

- Calculating entropy and information gain.
- Identifying the best feature and threshold for splitting the dataset.
- Visualizing the dataset and the decision boundary.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Example](#example)
- [Dependencies](#dependencies)

## Installation

Ensure you have Python installed along with the necessary libraries. You can install the required packages using pip:

```bash
pip install numpy matplotlib
```

## Usage

To run the script, execute the following command in your terminal. Make sure the dataset file `decision_tree_dataset.txt` is in the same directory.

```bash
python Decision_tree/decision_tree_preparations.py
```

## Functions

### `entropy(y: np.ndarray) -> float`

Calculates the entropy of a given target variable.

- **Parameters**:
  - `y`: A NumPy array of target class labels.
  
- **Returns**:
  - Entropy value as a float.

### `information_gain(y_parent: np.ndarray, index_split: np.ndarray) -> float`

Computes the information gain from a potential split.

- **Parameters**:
  - `y_parent`: A NumPy array of target class labels for the parent node.
  - `index_split`: A boolean NumPy array indicating the split.
  
- **Returns**:
  - Information gain as a float.

### `create_split(X: np.ndarray, split_dim: int, split_val: float) -> np.ndarray`

Creates a boolean index for splitting the dataset.

- **Parameters**:
  - `X`: A NumPy array of feature data.
  - `split_dim`: The dimension/index along which to split.
  - `split_val`: The threshold value for the split.
  
- **Returns**:
  - A boolean NumPy array indicating the split.

### `best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]`

Identifies the best split dimension and value based on information gain.

- **Parameters**:
  - `X`: A NumPy array of feature data.
  - `y`: A NumPy array of target class labels.
  
- **Returns**:
  - A tuple containing the best dimension index and the corresponding threshold value.

## Example

The script will load a dataset from `decision_tree_dataset.txt`, find the best split, and visualize the results.

1. Prepare your dataset in `decision_tree_dataset.txt` in CSV format:
    ```
    feature1, feature2, class
    1.0, 2.0, 0
    2.0, 3.0, 1
    ...
    ```

2. Run the script to see a plot showing the data points and the best decision boundary.

## Dependencies

This script requires the following Python libraries:
- `numpy`
- `matplotlib`
