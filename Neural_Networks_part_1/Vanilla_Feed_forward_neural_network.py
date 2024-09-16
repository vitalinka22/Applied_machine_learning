# -*- coding: utf-8 -*-
"""
Vanilla Feed forward neural network for solving the XOR case

(C) Merten Stender, TU Berlin
merten.stender@tu-berlin.de
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow import keras as keras
# import tensorflow as tf

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Overview on Keras models https://keras.io/api/models/

Dense layer: https://keras.io/api/layers/core_layers/dense/

Optimizers: https://keras.io/api/optimizers/

Losses: https://keras.io/api/losses/
"""

# create data for the XOR case
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([1, 0, 0, 1])

# build a neural network using TF's sequential API
model = Sequential()
# 1st hidden layer
model.add(Dense(2, input_shape=(2,), activation="sigmoid"))
# output layer
model.add(Dense(1, activation="sigmoid"))

# train the neural network
adam = keras.optimizers.Adam() #lr=0.05)
model.compile(loss="binary_crossentropy",
              optimizer=adam,
              )

# get a summary of the model
model.summary()

# train the neural network
history = model.fit(X, y, epochs=5000)

# make predictions
y_pred = model.predict(X)

fig = plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.yscale('log')
plt.show()

# iterate through all data samples and investigate truth vs. prediction
for _y, _y_pred in zip(y.flatten(), y_pred.flatten()):
    print(f'ground truth is {_y}, prediction is {float(_y_pred):.3f}')
