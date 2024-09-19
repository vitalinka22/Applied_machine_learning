# Neural Networks Part 1

## Overview

This project demonstrates the basics of a neural network implementation using NumPy. It includes weight initialization, forward propagation through a simple network, and a demonstration with a basic dataset.

## Files

- `Neural_Networks_1.py`: Implements a neural network with two hidden layers, performs forward propagation, and prints intermediate results.

## Functions

- `get_weights(input_shape, output_shape)`: Initializes a weight matrix with values set to 0.5.
- `get_bias(shape)`: Initializes a bias vector with values set to 0.5.
- `sigmoid(z: float)`: Computes the sigmoid activation function.
- `loss(y, y_pred)`: Computes the loss (note: the function contains an error; it should use `np.linalg.norm` instead of `np.linalg`).

## Usage

1. **Run the Script**

   Execute the script to perform forward propagation through a simple neural network:

   ```bash
   python Neural_Networks_1.py
   ```

   This script:
   - Initializes weights and biases.
   - Passes input data through a network with two hidden layers.
   - Prints intermediate values and final output.

2. **Details**

   - **Initialization**: Weights and biases are initialized with a constant value of 0.5.
   - **Forward Pass**: Computes output for each sample from the dataset using the sigmoid activation function.
   - **Dataset**: The script uses a simple XOR problem dataset.

## Example Output

The script prints:
- Intermediate values (`z` and `x`) for each layer.
- Final output for each input sample.

## Dependencies

- NumPy
