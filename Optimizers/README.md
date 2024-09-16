# Optimizers

## Overview

This project explores various optimization algorithms for minimizing a scalar loss function. It includes implementations of Gradient Descent (GD), Gradient Descent with Momentum (GDM), AdaGrad, and Adam optimization methods. The script visualizes the performance of these algorithms in minimizing the loss function.

## Files

- `Optimizers.py`: Contains the implementations of different optimization algorithms and generates plots to compare their performance.

## Functions

- `loss_scalar(theta_0: np.ndarray, theta_1: np.ndarray) -> float`:
  Computes the loss function value for given parameters \(\theta_0\) and \(\theta_1\).

- `loss(theta: np.ndarray) -> float`:
  Computes the loss function value for a given parameter vector `theta`. Supports both 1D and 2D arrays.

- `nabla_loss(theta: np.ndarray) -> np.ndarray`:
  Returns the gradient of the loss function with respect to the parameters.

- `gradient_descent_with_momentum(f, df, x0: np.ndarray, lr: float = 0.1, beta: float = 0.5)`:
  Implements the Gradient Descent with Momentum algorithm.

- `gradient_descent(f, df, x0: np.ndarray, lr: float = 0.1)`:
  Implements the standard Gradient Descent algorithm.

- `gradient_descent_adagrad(f, df, x0: np.ndarray, lr: float = 0.5)`:
  Implements the AdaGrad algorithm.

- `gradient_descent_adam(f, df, x0: np.ndarray, lr: float = 0.95, beta_1 = 0.9, beta_2 = 0.999)`:
  Implements the Adam optimization algorithm.

## Usage

1. **Run the Script**

   Execute the script to perform optimization and visualize the results:

   ```bash
   python Optimizers.py
   ```

2. **Visualizations**

   The script generates the following plots:
   - **Loss Surface**: A 3D plot showing the loss surface for the function.
   - **Loss Progression**: A plot of loss values over iterations for each optimization algorithm.
   - **Optimization Path**: A 3D plot and contour plot showing the paths taken by each optimization algorithm on the loss surface.

## Example Output

The script produces visualizations that help compare the convergence and performance of different optimization methods. It includes:
- A 3D surface plot of the loss function.
- Iteration-wise loss values for GD, GDM, AdaGrad, and Adam.
- 3D and contour plots illustrating the paths taken by each optimizer.

## Dependencies

- NumPy
- Matplotlib
