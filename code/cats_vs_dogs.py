# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:17:25 2018

@author: hehu
"""

# Import the network container and the three types of layers
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten

# Initialize the model
model = Sequential()

shape = (64, 64, 3)

# Add six convolutional layers. Maxpool after every second convolution.
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu",
input_shape=shape))
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(2, 2)) # Shrink feature maps to 32x32

model.add(Conv2D(filters=48, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=48, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(2, 2)) # Shrink feature maps to 16x16

model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(2, 2)) # Shrink feature maps to 8x8

# Vectorize the 8x8x64 representation to 4096x1 vector
model.add(Flatten())

# Add a dense layer with 128 nodes
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

# Finally, the output layer has 1 output with logistic sigmoid nonlinearity
model.add(Dense(1, activation="sigmoid"))

# Import the network container and the three types of layers
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense

# Initialize the VGG16 network. Omit the dense layers on top.
base_model = VGG16(include_top = False, weights = "imagenet",
input_shape = (64, 64, 3))

# We use the functional API, and grab the VGG16 output here:
w = base_model.output

# Now we can perform operations on w. First flatten it to 4096-dim vector:
w = Flatten()(w)

# Add dense layer:
w = Dense(128, activation = "relu")(w)

# Add output layer:
output = Dense(1, activation = "sigmoid")(w)

# Prepare the full model from input to output:
model = Model(inputs = [base_model.input], outputs = [output])

# Also set the last Conv block (3 layers) as trainable.
# There are four layers above this block, so our indices
# start at -5 (i.e., last minus five):
model.layers[-5].trainable = True
model.layers[-6].trainable = True
model.layers[-7].trainable = True
