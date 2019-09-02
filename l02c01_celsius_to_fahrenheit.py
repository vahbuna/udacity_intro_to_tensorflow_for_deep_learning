# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The problem we will solve is to convert from Celsius to Fahrenheit, where the approximate formula is:
    f = c * 1.8 + 32
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Set up training data
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

# Create the model
def l1_model():
    # Build a layer
    # input_shape=[1] - This specifies that the input to this layer is a single value. That is, the shape is a one-dimensional array with one member.
    # units=1 - This specifies the number of neurons in the layer.
    l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

    # Assemble layers into the model
    return tf.keras.Sequential([l0])

model = l1_model()

# Compile the model, with loss and optimizer functions
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# Train the model
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

# Display training statistics
# We can use history object to plot how the loss of our model goes down after each training epoch.
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.savefig('l02c01_plot.png')

# Use the model to predict values
print(model.predict([100.0]))

# Looking at the layer weights
print("These are the layer variables: {}".format(model.layers[0].get_weights()))

# The first variable is close to ~1.8 and the second to ~32.

# A little experiment
# Just for fun, what if we created more Dense layers with different units, which therefore also has more variables?
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0]))
print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))
print("These are the l0 variables: {}".format(l0.get_weights()))
print("These are the l1 variables: {}".format(l1.get_weights()))
print("These are the l2 variables: {}".format(l2.get_weights()))
#  But when you look at the variables (weights) in the l0 and l1 layers, they are nothing even close to ~1.8 and ~32.
# The added complexity hides the "simple" form of the conversion equation.
