import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define the model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Define the training data
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

# Train the model
model.fit(xs, ys, epochs=1000)

# Make a prediction
print(model.predict(np.array([7.0])))
