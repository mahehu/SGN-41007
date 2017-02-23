# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 15:59:14 2016

@author: hehu
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import pickle

def generate_data(N):

    X1 = np.random.randn(2, N)
    X2 = np.random.randn(2, N)

    M1 = np.array([[1.5151, -0.1129], [0.1399, 0.6287]])
    M2 = np.array([[0.8602, 1.2461], [-0.0737, -1.5240]])

    T1 = np.array([-1, 1]).reshape((2, 1))
    T2 = np.array([-5, 2]).reshape((2, 1))

    X1 = np.dot(M1, X1) + np.tile(T1, [1, N])
    X2 = np.dot(M2, X2) + np.tile(T2, [1, N])

    X1 = X1[::-1,:]
    X2 = X2[::-1,:]

    return X1, X2

def grad(w, X, y):

    G = 0

    for n in range(X.shape[0]):
        numerator = np.exp(-y[n] * np.dot(w, X[n])) * (-y[n]) * X[n]
        denominator = 1 + np.exp(-y[n] * np.dot(w, X[n]))

        G += numerator / denominator

    return G

def loss (w, X, y):

    if len(w.shape) == 2:
        Y = np.tile(y, (w.T.shape[1], 1)).T
    else:
        Y = y

    L = np.sum(1 + np.exp(-Y * np.dot(X, w.T)), axis = 0)
    return L

def log_loss(w, X, y):

    if len(w.shape) == 2:
        Y = np.tile(y, (w.T.shape[1], 1)).T
    else:
        Y = y

    L = np.sum(np.log(1 + np.exp(-Y * np.dot(X, w.T))), axis = 0)
    return L

if __name__ == "__main__":

    plt.close("all")

#    matplotlib.rcdefaults()
#    font = {'family' : 'sans-serif'}
#    matplotlib.rc('font', **font)

    # Generate random training data

    N = 200
    np.random.seed(1)

    X1, X2 = generate_data(N)

    X = np.concatenate((X1.T, X2.T))
    y = np.concatenate((np.ones(N), -1 * np.ones(N)))
    idx = range(y.size)
    random.shuffle(idx)

    X = X[idx,:]
    y = y[idx]

    X[:, 0] = X[:, 0] - X[:, 0].mean()
    X[:, 1] = X[:, 1] - X[:, 1].mean()

    with open("log_loss_data.pkl", "w") as fp:
        pickle.dump({'X': X, 'y': y}, fp)

    w = np.array([1, -1])

    step_size = 0.001
    W = []
    accuracies = []
    losses = []

    for iteration in range(100):
        w = w - step_size * grad(w, X, y)

        loss_val = log_loss(w, X, y)
        print loss_val
        print ("Iteration %d: w = %s (log-loss = %.2f)" % \
              (iteration, str(w), loss_val))

        # Predict class 1 probability
        y_prob = 1 / (1 + np.exp(-np.dot(X, w)))
                # Threshold at 0.5 (results are 0 and 1)
        y_pred = (y_prob > 0.5).astype(int)
                # Transform [0,1] coding to [-1,1] coding
        y_pred = 2*y_pred - 1

        accuracy = np.mean(y_pred == y)
        accuracies.append(accuracy)
        losses.append(loss)

        W.append(w)

    W = np.array(W)

    # Plot the path.

    fig, ax = plt.subplots(2, 1, figsize = [5, 5])

    xmin = -1
    xmax = 1.5
    ymin = -1
    ymax = 4

    Xg, Yg = np.meshgrid(np.linspace(xmin, xmax, 100),
                         np.linspace(ymin, ymax, 100))
    Wg = np.vstack([Xg.ravel(), Yg.ravel()]).T
    Zg = log_loss(Wg, X, y)
    Zg = np.reshape(Zg, Xg.shape)
    levels = np.linspace(70, 100, 20)
    
    ax[0].contourf(Xg, Yg, Zg,
                  alpha=0.5,
                  cmap=plt.cm.bone,
                  levels = levels)

    ax[0].plot(W[:, 0], W[:, 1], 'ro-')
    ax[0].set_xlabel('w$_0$')
    ax[0].set_ylabel('w$_1$')
    ax[0].set_title('Optimization path')
    ax[0].grid()
    ax[0].axis([xmin, xmax, ymin, ymax])

    ax[0].annotate('Starting point',
             xy=(W[0, 0], W[0, 1]),
             xytext=(-0.5, 0),
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=0.15",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center',
             verticalalignment='middle')


    ax[0].annotate('Endpoint',
             xy=(W[-1, 0], W[-1, 1]),
             xytext=(1.2, 0),
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=0.15",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center',
             verticalalignment='middle')

    ax[1].plot(100.0 * np.array(accuracies), linewidth = 2,
               label = "Classification Accuracy")
    ax[1].set_ylabel('Accuracy / %')
    ax[1].set_xlabel('Iteration')
    ax[1].legend(loc = 4)
    ax[1].grid()
    plt.tight_layout()
    plt.savefig("log_loss_minimization.pdf", bbox_inches = "tight")
