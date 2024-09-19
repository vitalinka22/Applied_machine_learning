# Linear Regression Analysis

## Overview

This project demonstrates linear regression using NumPy to model driving power data, specifically focusing on calculating rolling resistance.

## Files

- `lin_regress.py`: Implements linear regression and evaluates driving power data.
- `driving_data.csv`: Dataset for analyzing driving power.

## Usage

1. **Run Linear Regression**

   Execute the script to perform linear regression on the example data:

   ```bash
   python lin_regress.py
   ```

   This script:
   - Performs linear regression to model driving power.
   - Visualizes data and regression results.

2. **Key Functions**

   - `lin_regress(x: np.array, y: np.array) -> tuple[float, float]`: Performs linear regression and returns coefficients.
   - `wind_resistance(v: np.ndarray) -> np.ndarray`: Calculates wind resistance force.
   
3. **Steps in the Script**

   - **Data Preparation**: Load and preprocess the driving data.
   - **Plot Data**: Visualize initial driving power vs. velocity.
   - **Calculate Rolling Resistance**: Subtract wind resistance and plot power without wind.
   - **Linear Regression**: Fit a linear model to estimate rolling resistance.

4. **Results**

   - **Model**: Displays linear regression results.
   - **Plots**: Visualizes the relationship between velocity, power, and rolling resistance.

## Requirements

- NumPy
- Matplotlib
- scikit-learn (for advanced use)
