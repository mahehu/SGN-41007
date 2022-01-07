# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:58:51 2016

@author: hehu
"""

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV
#from sklearn.linear_model import RandomizedLogisticRegression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys

if __name__ == "__main__":
        
    digits = load_digits()
    
    numDigits = 10
    plt.figure(figsize=[4, 12])
    images_and_labels = list(zip(digits.images, digits.target))
    
    for index, (image, label) in enumerate(images_and_labels[:numDigits]):
        plt.subplot(numDigits/2, 2, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Class %i' % label)

    #plt.tight_layout()
    plt.savefig("../images/digits.pdf", bbox_inches = "tight")
    
    idx = (digits.target == 8) | (digits.target == 9)
    X = digits.data[idx, :]
    y = digits.target[idx]
    
    ss = StandardScaler()
    X = ss.fit_transform(X)
    
    skf = StratifiedKFold(y, 5, shuffle = True, random_state = 1)
    rfecv = RFECV(estimator=LinearDiscriminantAnalysis(), cv = skf)
    rfecv.fit(X, y)
    
    plt.figure()
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, linewidth = 2)

    topScore = np.max(rfecv.grid_scores_)
    n_feat = np.sum(rfecv.support_)
    plt.annotate('Top score %.1f %% with %d features' % (100.0*topScore, n_feat), 
             xy=(n_feat, topScore), 
             xytext=(35, 0.92), 
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

    plt.annotate('All features: %.1f %%' % (100.0*rfecv.grid_scores_[-1]), 
             xy=(len(rfecv.grid_scores_), rfecv.grid_scores_[-1]), 
             xytext=(55, 0.95), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='middle')
             
    plt.savefig("../images/rfe_accuracy.pdf", bbox_inches = "tight")
    
    mask = rfecv.support_.reshape(8, 8)
    plt.figure()
    plt.imshow(mask.astype(int), interpolation = 'nearest')
    plt.title("RFECV: Selected %d features (bright = selected)" % np.count_nonzero(mask))
    plt.savefig("../images/rfe_mask.pdf", bbox_inches = "tight")
    
    ############ LR
    
    lr = LogisticRegression(penalty = "l1")
    C_range = 10.0 ** np.arange(-5,12, 0.5)
    
    accuracies = []
    nonzeros = []
    bestScore = 0
    
    skf = StratifiedKFold(y, 5, shuffle = True, random_state = 3)
    np.random.seed(10)
    
    for C in C_range:
        lr.C = C
        score = cross_val_score(lr, X, y, cv = skf).mean()
        lr.fit(X, y)
        nonzeros.append(np.count_nonzero(lr.coef_))
        accuracies.append(score)
        
        if score > bestScore:
            bestScore = score
            bestCoef = lr.coef_.ravel()
            bestC = C
            
    plt.figure()
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.plot(nonzeros, accuracies, linewidth = 2)
    topIdx = np.argmax(accuracies)
    topScore = np.max(accuracies)
    top_nz = nonzeros[topIdx]
    
    plt.annotate('Top score %.1f %% with %d features' % (100.0*topScore, top_nz), 
             xy=(top_nz, topScore), 
             xytext=(25, 0.65), 
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

    plt.annotate('All features: %.1f %%' % (100.0*accuracies[-1]), 
             xy=(nonzeros[-1], accuracies[-1]), 
             xytext=(45, 0.75), 
             size=13,
             bbox=dict(boxstyle="round4", fc="w", ec = "g"),
             arrowprops=dict(arrowstyle="simple",
                             connectionstyle="arc3,rad=0.2",
                             shrinkA = 0,
                             shrinkB = 8,
                             fc = "g",
                             ec = "g"),
             horizontalalignment='center', 
             verticalalignment='middle')
             
    plt.savefig("../images/L1_digits_accuracy.pdf", bbox_inches = "tight")
    
    mask = (bestCoef != 0).astype(np.uint8)
    mask = mask.reshape(8, 8)
    plt.figure()
    plt.imshow(mask.astype(int), interpolation = 'nearest')
    plt.title("LOGREG: Selected %d features (bright = selected)" % np.count_nonzero(mask))
    plt.savefig("../images/L1_digits_mask.pdf", bbox_inches = "tight")
    
    # Stability selection
#    
#    model = RandomizedLogisticRegression(C = 1, 
#                                         sample_fraction = 0.75,
#                                         n_resampling = 1000,
#                                         selection_threshold=0.25,
#                                         n_jobs = 2,
#                                         random_state = 42)
#                            
#    model.fit(X, y)
#    mask = model.get_support()
#    mask = mask.reshape(8,8)

  #  plt.figure()
  #  plt.imshow(mask.astype(int), interpolation = 'nearest')
  #  plt.title("RLR: Selected %d features\n(bright = selected)" % np.count_nonzero(mask))
  #  plt.savefig("../images/Stability_selection_digits_mask.pdf", bbox_inches = "tight")
    
    plt.figure()
    plt.imshow(rfecv.ranking_.max() - rfecv.ranking_.reshape(8,8))
    plt.title("Order of RFECV selection\n(dark = dropped first)")
    plt.savefig("../images/RVECF_ranking.pdf", bbox_inches = "tight")

    plt.figure()    
    rf = RandomForestClassifier(n_estimators = 500)
    rf.fit(X, y)
    plt.imshow(rf.feature_importances_.reshape(8,8))
    plt.title("Random forest feature importances\n(bright = important)")
    plt.savefig("../images/RF_ranking.pdf", bbox_inches = "tight")

#    plt.figure()    
#    plt.imshow(model.scores_.reshape(8,8))
#    plt.title("Randomized logistic regression scores\n(bright = important)")
#    plt.savefig("../images/RLR_ranking.pdf", bbox_inches = "tight")
    
    