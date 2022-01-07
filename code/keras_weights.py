# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:45:20 2016

@author: hehu
"""

from keras.models import model_from_yaml
import matplotlib.pyplot as plt
import numpy as np
import theano
from keras import backend as K
from keras.datasets import mnist

if __name__ == "__main__":
    
    with open("mnist_classifier.yaml", "r") as fp:
        model = model_from_yaml(fp.read())
        
    model.load_weights('mnist_classifier.h5')
    
    # Visualize Layer 1
    
    weights = model.layers[0].get_weights()[0]

    fig, ax = plt.subplots(nrows = 8, ncols = 4, figsize = [5,10])
    
    for k in range(4):
        for j in range(8):
            w = weights[k + j*4,0,...]
            ax[j,k].imshow(w, interpolation='none', cmap=plt.get_cmap('gray'))
            ax[j,k].xaxis.set_ticks([])
            ax[j,k].xaxis.set_ticks_position('none') 
            ax[j,k].yaxis.set_ticks([])
            ax[j,k].yaxis.set_ticks_position('none') 
            
    plt.tight_layout()
    plt.savefig("../images/keras_l1_filters.pdf", bbox_axes = "tight")

    # Extract layer outputs:
    
    # input image dimensions
    img_rows, img_cols = 28, 28
    
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    get_layer_output = K.function([model.layers[0].input],
                                      [model.layers[0].get_output(train=False)])
    get_layer_2_output = K.function([model.layers[0].input],
                                      [model.layers[1].get_output(train=False)])
    get_layer_3_output = K.function([model.layers[0].input],
                                      [model.layers[2].get_output(train=False)])
    get_layer_4_output = K.function([model.layers[0].input],
                                      [model.layers[3].get_output(train=False)])
    get_layer_5_output = K.function([model.layers[0].input],
                                      [model.layers[4].get_output(train=False)])
                         
     ### Layer 1 VISUALIZATION
                         
    layer_output = get_layer_output([X_test[np.newaxis,0,...]])[0]
    
    fig, ax = plt.subplots(nrows = 8, ncols = 4, figsize = [5,10])
    
    for k in range(4):
        for j in range(8):
            w = layer_output[0, k + j*4,...]
            w = w - w.min()
            w = (255 * w / w.max()).astype(np.uint8)
            ax[j,k].imshow(w, interpolation='none', cmap=plt.get_cmap('gray'))#, cmap=plt.get_cmap('gray'))
            ax[j,k].xaxis.set_ticks([])
            ax[j,k].xaxis.set_ticks_position('none') 
            ax[j,k].yaxis.set_ticks([])
            ax[j,k].yaxis.set_ticks_position('none') 
            
    plt.tight_layout()
    plt.savefig("../images/keras_l1_outputs.pdf", bbox_axes = "tight")

    layer_output = get_layer_output([X_test[np.newaxis,1,...]])[0]
    
    fig, ax = plt.subplots(nrows = 8, ncols = 4, figsize = [5,10])
    
    for k in range(4):
        for j in range(8):
            w = layer_output[0, k + j*4,...]
            w = w - w.min()
            w = (255 * w / w.max()).astype(np.uint8)
            ax[j,k].imshow(w, interpolation='none', cmap=plt.get_cmap('gray'))#, cmap=plt.get_cmap('gray'))
            ax[j,k].xaxis.set_ticks([])
            ax[j,k].xaxis.set_ticks_position('none') 
            ax[j,k].yaxis.set_ticks([])
            ax[j,k].yaxis.set_ticks_position('none') 
            
    plt.tight_layout()
    plt.savefig("../images/keras_l1_outputs_2.pdf", bbox_axes = "tight")
    
    plt.figure()
    plt.imshow(X_test[0,0,...], interpolation='none', cmap=plt.get_cmap('gray'))
    plt.savefig("../images/keras_l1_input.pdf", bbox_axes = "tight")

    plt.imshow(X_test[1,0,...], interpolation='none', cmap=plt.get_cmap('gray'))
    plt.savefig("../images/keras_l1_input_2.pdf", bbox_axes = "tight")

    ### Layer 2 VISUALIZATION

    layer_output = get_layer_3_output([X_test[np.newaxis,0,...]])[0]
    
    fig, ax = plt.subplots(nrows = 8, ncols = 4, figsize = [5,10])
    
    for k in range(4):
        for j in range(8):
            w = layer_output[0, k + j*4,...]
            w = w - w.min()
            w = (255 * w / w.max()).astype(np.uint8)
            ax[j,k].imshow(w, interpolation='none', cmap=plt.get_cmap('gray'))#, cmap=plt.get_cmap('gray'))
            ax[j,k].xaxis.set_ticks([])
            ax[j,k].xaxis.set_ticks_position('none') 
            ax[j,k].yaxis.set_ticks([])
            ax[j,k].yaxis.set_ticks_position('none') 
            
    plt.tight_layout()
    plt.savefig("../images/keras_l2_outputs.pdf", bbox_axes = "tight")

    layer_output = get_layer_3_output([X_test[np.newaxis,1,...]])[0]
    
    fig, ax = plt.subplots(nrows = 8, ncols = 4, figsize = [5,10])
    
    for k in range(4):
        for j in range(8):
            w = layer_output[0, k + j*4,...]
            w = w - w.min()
            w = (255 * w / w.max()).astype(np.uint8)
            ax[j,k].imshow(w, interpolation='none', cmap=plt.get_cmap('gray'))#, cmap=plt.get_cmap('gray'))
            ax[j,k].xaxis.set_ticks([])
            ax[j,k].xaxis.set_ticks_position('none') 
            ax[j,k].yaxis.set_ticks([])
            ax[j,k].yaxis.set_ticks_position('none') 
            
    plt.tight_layout()
    plt.savefig("../images/keras_l2_outputs_2.pdf", bbox_axes = "tight")
    
    plt.show()
    
    