
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

if __name__ == "__main__":
    
    batch_size = 128
    nb_classes = 10
    nb_epoch = 10
    
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Add a dummy color channel dimension
    # Data has to be 4D: (sample_id, color_channel, y, x)
    # Now it is 3D: (sample_id, y, x)

    X_train = X_train[:, np.newaxis, ...]
    X_test  = X_test[:, np.newaxis, ...]

    print("Training data shape is ", X_train.shape)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # Prepare model

    model = Sequential()
    
    num_featmaps = 32   # This many filters per layer
    num_classes = 10    # Digits 0,1,...,9
    num_epochs = 50     # Show all samples 50 times
    w, h = 5, 5         # Conv window size

    # Layer 1: needs input_shape as well.

    model.add(Convolution2D(num_featmaps, w, h,
                            input_shape=(1, 28, 28), 
                            activation = 'relu'))
    
    # Layer 2:

    model.add(Convolution2D(num_featmaps, w, h, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Layer 3: dense layer with 128 nodes
    # Flatten() vectorizes the data:
    # 32x10x10 -> 3200 
    # (10x10 instead of 14x14 due to border effect)

    model.add(Flatten()) 
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    
    # Layer 4: Last layer producing 10 outputs.
    model.add(Dense(num_classes, activation='softmax'))

    # Compile and run

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics = ['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    
