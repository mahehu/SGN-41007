# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 13:43:24 2015

@author: hehu
"""

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.special import erf
from twoClassExample import generate_data, visualize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def gaussian(x, mu, sig):
    multiplier = (1/np.sqrt(2*np.pi*sig**2.))
    return multiplier * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian_log(x, mu, sig):
    multiplier = (1/np.sqrt(2*np.pi*sig**2.))
    return np.log(multiplier) + (-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
def gaussian_cdf(x, mu, sigma):
    
    p = (1.0 / 2.0) * (1.0 + erf((x - mu) / (sigma * np.sqrt(2))))
    return p

def plot_border(coef, intercept):

    coef = coef.ravel()
    intercept = intercept.ravel()
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    print(xmin, xmax)
    
    x = np.linspace(xmin, xmax, 100)
    plt.plot(x, -coef[0] / coef[1] * x - intercept / coef[1], 'k--', linewidth = 2)
             
if __name__ == "__main__":
    
    matplotlib.rcdefaults()
    font = {'family' : 'sans-serif'}
    matplotlib.rc('font', **font)
    
    plt.close("all")
    
    x = np.linspace(-4, 4, 100)
    y0 = gaussian(x, 0, 1)
    y1 = gaussian(x, 1, 1)
    
    ax1 = plt.subplot(211)
    plt.plot(x, y0, linewidth = 2, label = "p(x[0] | ${\cal H}_0$)")
    plt.plot(x, y1, 'r-', linewidth = 2, label = "p(x[0] | ${\cal H}_1$)")
    ax1.set_xlabel('x[0]')
    ax1.set_ylabel('Likelihood')
    plt.legend(loc = 2)
    plt.title("Likelihood of observing different values of x[0] given ${\cal H}_0$ or ${\cal H}_1$")
    plt.savefig("../images/NeymanPearson.pdf", bbox_inches = "tight")
    
    #######################################################
    # errorTypes1.pdf
    
    plt.close("all")
    
    x = np.linspace(-4, 4, 400)
    y0 = gaussian(x, 0, 1)
    y1 = gaussian(x, 1, 1)
    
    ax1 = plt.subplot(211)
    ax1.plot(x, y0, linewidth = 2, label = "p(x[0] | ${\cal H}_0$)")
    ax1.plot(x, y1, 'r-', linewidth = 2, label = "p(x[0] | ${\cal H}_1$)")
    ax1.fill_between(x, y1, where = (x <= 0.5), facecolor='red', alpha=0.5)
    ax1.fill_between(x, y0, where = (x >= 0.5), facecolor='blue', alpha=0.5)
    
    ax1.annotate('Decide ${\cal H_0}$ when ${\cal H_1}$ holds', 
             xy=(-0.5, 0.025), 
             xytext=(-7, 0.25), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=+0.15",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')
    
    ax1.annotate('Decide ${\cal H_1}$ when ${\cal H_0}$ holds', 
             xy=(1.5, 0.025), 
             xytext=(7, 0.25), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.15",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')             
            
            
    ax1.set_xlabel('x[0]')
    ax1.set_ylabel('Likelihood')
    #plt.legend(loc = 2)
    
    plt.savefig("../images/errorTypes1.pdf", bbox_inches = "tight")
    
    #######################################################
    # errorTypes2.pdf
    
    plt.close("all")
    
    x = np.linspace(-4, 4, 400)
    y0 = gaussian(x, 0, 1)
    y1 = gaussian(x, 1, 1)
    
    ax1 = plt.subplot(211)
    ax1.plot(x, y0, linewidth = 2, label = "p(x[0] | ${\cal H}_0$)")
    ax1.plot(x, y1, 'r-', linewidth = 2, label = "p(x[0] | ${\cal H}_1$)")
    ax1.fill_between(x, y1, where = (x <= 0.), facecolor='red', alpha=0.5)
    ax1.fill_between(x, y0, where = (x >= 0.), facecolor='blue', alpha=0.5)
    
    ax1.annotate('Decide ${\cal H_0}$ when ${\cal H_1}$ holds', 
             xy=(-0.5, 0.025), 
             xytext=(-7, 0.25), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=+0.15",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')
    
    ax1.annotate('Decide ${\cal H_1}$ when ${\cal H_0}$ holds', 
             xy=(1.5, 0.025), 
             xytext=(7, 0.25), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.15",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')             
            
            
    ax1.set_xlabel('x[0]')
    ax1.set_ylabel('Likelihood')
    plt.title("Detection threshold at 0. Small amount of missed\ndetections (red) but many false matches (blue).")
    #plt.legend(loc = 2)
    
    plt.savefig("../images/errorTypes2.pdf", bbox_inches = "tight")
    
        #######################################################
    # errorTypes1.pdf
    
    plt.close("all")
    
    x = np.linspace(-4, 4, 400)
    y0 = gaussian(x, 0, 1)
    y1 = gaussian(x, 1, 1)
    
    ax1 = plt.subplot(211)
    ax1.plot(x, y0, linewidth = 2, label = "p(x[0] | ${\cal H}_0$)")
    ax1.plot(x, y1, 'r-', linewidth = 2, label = "p(x[0] | ${\cal H}_1$)")
    ax1.fill_between(x, y1, where = (x <= 1.5), facecolor='red', alpha=0.5)
    ax1.fill_between(x, y0, where = (x >= 1.5), facecolor='blue', alpha=0.5)
    
    ax1.annotate('Decide ${\cal H_0}$ when ${\cal H_1}$ holds', 
             xy=(-0.5, 0.025), 
             xytext=(-7, 0.25), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=+0.15",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')
    
    ax1.annotate('Decide ${\cal H_1}$ when ${\cal H_0}$ holds', 
             xy=(2, 0.015), 
             xytext=(7, 0.25), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')             
            
            
    ax1.set_xlabel('x[0]')
    ax1.set_ylabel('Likelihood')
    
    plt.title("Detection threshold at 1.5. Small amount of false\nmatches (blue) but many missed detections (red).")
    #plt.legend(loc = 2)
    
    plt.savefig("../images/errorTypes3.pdf", bbox_inches = "tight")
    
    # compute threshold such that P_FS = 0.1
    T = stats.norm.isf(0.1, loc = 0, scale = 1)
    print(T)
    
    # Plot ROC curve for 2 gaussians (mu = 0 and mu = 1; std = 1)
    
    plt.close("all")
    
    gamma_range = np.linspace(-10,10,1000)
    points = []
    
    for gamma in gamma_range:
        PD  = gaussian_cdf(gamma, mu = 0, sigma = 1)
        PFA = gaussian_cdf(gamma, mu = 1, sigma = 1)
        points.append([PD, PFA])
    
    points = np.array(points)
    plt.plot(points[:, 1], points[:, 0], linewidth = 2)    
    plt.xlabel('Probability of False Alarm $P_{FA}$')
    plt.ylabel('Probability of Detection $P_{D}$')
    plt.annotate('Small $\gamma$\n(high sensitivity)', 
             xy=(0.98, 0.9), 
             xytext=(0.6, 0.7), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.1",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')             

    plt.annotate('Large $\gamma$\n(low sensitivity)', 
             xy=(0.13, 0.02), 
             xytext=(0.3, 0.4), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=0.16",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')             
   # plt.show()
    plt.title("ROC Curve")
    plt.savefig("../images/RocCurve.pdf", bbox_inches = "tight")
    
    
   # Plot ROC curves for different sigmas.
    
    plt.close("all")
    
    gamma_range = np.linspace(-10,10,1000)
    sigma_range = np.arange(0.2, 1.01, 0.2)
    
    for sigma in sigma_range:
        points = []
        
        for gamma in gamma_range:
            PD  = gaussian_cdf(gamma, mu = 0, sigma = sigma)
            PFA = gaussian_cdf(gamma, mu = 1, sigma = sigma)
            points.append([PD, PFA])
        
        points = np.array(points)
        auc = np.trapz(points[:, 0], points[:, 1])
        plt.plot(points[:, 1], points[:, 0], linewidth = 2, label = "$\sigma$ = %.1f (AUC = %.2f)" % (sigma, auc))    

    plt.plot([0, 1], [0, 1], '--', linewidth = 2, label = "Random Guess (AUC = 0.5)")
    plt.legend(loc = 4)
    plt.xlabel('Probability of False Alarm $P_{FA}$')
    plt.ylabel('Probability of Detection $P_{D}$')

    #plt.show()
    #plt.title("ROC Curve")
    plt.savefig("../images/RocCurve2.pdf", bbox_inches = "tight")
    
    # Empirical ROC
    
    plt.close("all")
        
    # Generate random training data

    N = 200
    np.random.seed(2015)
    
    X1, X2 = generate_data(N)
    
    X = np.concatenate((X1.T, X2.T))
    y = np.concatenate((np.ones(N), np.zeros(N)))    
    
    # Generate test sample
    
    np.random.seed(2016)
    
    X1_test, X2_test = generate_data(N)
    
    X_test = np.concatenate((X1_test.T, X2_test.T))
    y_test = np.concatenate((np.ones(N), np.zeros(N)))    
    
    visualize(None, X, y, None)
    plt.title("Training Data")

    plt.savefig("../images/TrainingData_2class.pdf", bbox_inches = "tight")
    
    visualize(None, X_test, y_test, None)
    plt.title("Test Data")
    plt.savefig("../images/TestData_2class.pdf", bbox_inches = "tight")
    
    clf = LogisticRegression()
    clf.fit(X, y)        
    
    plot_border(clf.coef_, clf.intercept_)
    plt.title("Classifier with minimum error boundary")

    plt.annotate('Class Boundary', 
             xy=(3, -3.7), 
             xytext=(3.5, -1.5), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')             

    err1 = np.count_nonzero(clf.predict(X_test[y_test == 0, :]) == 1)
    err2 = np.count_nonzero(clf.predict(X_test[y_test == 1, :]) == 0)
    
    err1 = 100.0 * err1 / np.count_nonzero(y_test == 0)
    err2 = 100.0 * err2 / np.count_nonzero(y_test == 1)

    plt.annotate('%.1f %% of circles\ndetected as cross' % err2, 
             xy=(0.3, -4.4), 
             xytext=(0, -9), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')        

    plt.annotate('%.1f %% of crosses\ndetected as circle' % err1, 
             xy=(-0.4, -0.85), 
             xytext=(-0.2,1), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')        
             
    plt.savefig("../images/2classBoundary.pdf", bbox_inches = "tight")
    
    plt.close("all")
    visualize(None, X_test, y_test, None)
    clf.intercept_ += 2
    plot_border(clf.coef_, clf.intercept_)

    plt.annotate('Class Boundary', 
             xy=(3, -4.7), 
             xytext=(3.5, -1.5), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')             

    err1 = np.count_nonzero(clf.predict(X_test[y_test == 0, :]) == 1)
    err2 = np.count_nonzero(clf.predict(X_test[y_test == 1, :]) == 0)
    
    err1 = 100.0 * err1 / np.count_nonzero(y_test == 0)
    err2 = 100.0 * err2 / np.count_nonzero(y_test == 1)

    plt.annotate('%.1f %% of circles\ndetected as cross' % err2, 
             xy=(0.3, -4.4), 
             xytext=(0, -9), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')        

    plt.annotate('%.1f %% of crosses\ndetected as circle' % err1, 
             xy=(-0.4, -0.85), 
             xytext=(-0.2,1), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='center')        
             
    plt.title("Classifier with boundary lifted down")

    plt.savefig("../images/2classBoundary_highSensitivity.pdf", bbox_inches = "tight")
        
    plt.close("all")
    visualize(None, X_test, y_test, None)
    clf.intercept_ -= 4
    plot_border(clf.coef_, clf.intercept_)

    plt.title("Classifier with boundary lifted up")

    plt.annotate('Class Boundary', 
             xy=(3, -2.4), 
             xytext=(3.5, -1.5), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='middle')             

    err1 = np.count_nonzero(clf.predict(X_test[y_test == 0, :]) == 1)
    err2 = np.count_nonzero(clf.predict(X_test[y_test == 1, :]) == 0)
    
    err1 = 100.0 * err1 / np.count_nonzero(y_test == 0)
    err2 = 100.0 * err2 / np.count_nonzero(y_test == 1)

    plt.annotate('%.1f %% of circles\ndetected as cross' % err2, 
             xy=(0.3, -4.4), 
             xytext=(0, -9), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='middle')        

    plt.annotate('%.1f %% of crosses\ndetected as circle' % err1, 
             xy=(-0.4, -0.85), 
             xytext=(-0.2,1), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='middle')        
             
    plt.savefig("../images/2classBoundary_lowSensitivity.pdf", bbox_inches = "tight")
    
    gammaRange = np.linspace(0, 1, 1000)
    
    ROC = []
    
    clf = LogisticRegression()
    clf.fit(X, y)

    for gamma in gammaRange:
        
        err1 = np.count_nonzero(clf.predict_proba(X_test[y_test == 0, :])[:,1] <= gamma)
        err2 = np.count_nonzero(clf.predict_proba(X_test[y_test == 1, :])[:,1] > gamma)
    
        err1 = 1.0 * err1 / np.count_nonzero(y_test == 0)
        err2 = 1.0 * err2 / np.count_nonzero(y_test == 1)
        
        ROC.append([err1, err2])
        
    plt.close("all")
    ROC = np.array(ROC)
    
    ROC = ROC[::-1, :]
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    
    plt.plot(1-ROC[:, 0], ROC[:, 1], linewidth = 2, label="AUC = %.2f" % (auc))
    plt.legend(loc = 4)
    #plt.savefig("../images/2classBoundary_ROC.pdf", bbox_inches = "tight")
    
    plt.plot([0.135], [0.94], "ro")
    plt.annotate('Figure on the Right', 
             xy=(0.135, 0.93), 
             xytext=(0.4,0.4), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=-0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='middle')        
    
    plt.ylabel("Probability of Detection $P_D$")
    plt.xlabel("Probability of False Alarm $P_{FA}$")
    plt.savefig("../images/2classBoundary_ROC.pdf", bbox_inches = "tight")
    
    plt.close("all")
    
    classifiers = [(LogisticRegression(), "Logistic Regression"),
                   (SVC(probability = True), "Support Vector Machine"),
                   (RandomForestClassifier(n_estimators = 100), "Random Forest"),
                   (KNeighborsClassifier(), "Nearest Neighbor")]
            
    for clf, name in classifiers:
        clf.fit(X, y)
        
        ROC = []
    
        for gamma in gammaRange:
            
            err1 = np.count_nonzero(clf.predict_proba(X_test[y_test == 0, :])[:,1] <= gamma)
            err2 = np.count_nonzero(clf.predict_proba(X_test[y_test == 1, :])[:,1] > gamma)
        
            err1 = 1.0 * err1 / np.count_nonzero(y_test == 0)
            err2 = 1.0 * err2 / np.count_nonzero(y_test == 1)
            
            ROC.append([err1, err2])
        ROC = np.array(ROC)
        
        ROC = ROC[::-1, :]
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
        
        plt.plot(1-ROC[:, 0], ROC[:, 1], linewidth = 2, label="%s (AUC = %.2f)" % (name, auc))
    
    plt.legend(loc = 4)
    plt.ylabel("Probability of Detection $P_D$")
    plt.xlabel("Probability of False Alarm $P_{FA}$")
    plt.savefig("../images/2classBoundary_ROC_4clf.pdf", bbox_inches = "tight")
    
    plt.close("all")
    