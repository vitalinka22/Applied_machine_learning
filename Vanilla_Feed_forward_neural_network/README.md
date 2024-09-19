# Vanilla Feed-Forward Neural Network

## Overview

This project demonstrates a simple feed-forward neural network using TensorFlow and Keras to solve the XOR problem.

## Features

- **Data**: Synthetic XOR dataset.
- **Model**: 1 hidden layer, 1 output layer.
- **Training**: 5000 epochs with binary crossentropy loss.
- **Evaluation**: Loss plot and predictions comparison.

## Code Highlights

- **Data Preparation**:
  ```python
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  y = np.array([1, 0, 0, 1])
  ```

- **Model Definition**:
  ```python
  model = Sequential([
      Dense(2, activation="sigmoid", input_shape=(2,)),
      Dense(1, activation="sigmoid")
  ])
  ```

- **Training**:
  ```python
  history = model.fit(X, y, epochs=5000)
  ```

- **Evaluation**:
  ```python
  plt.plot(history.history['loss'])
  plt.show()
  ```

## Dependencies

- TensorFlow
- NumPy
- Matplotlib

Install required packages with:
```bash
pip install tensorflow numpy matplotlib
```

## Usage

Run the script:
```bash
python Vanilla_Feed_forward_neural_network.py
```
