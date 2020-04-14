""" A helper to plot decision boundary of a two class classifier"""

import numpy as np 
import matplotlib.pyplot as plt 

def plot_decision_boundaries(clf, X, y):
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),
                        np.arange(y_min, y_max, 0.01))

    H = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    H = H.reshape(xx.shape)
    cm = plt.cm.RdBu
    plt.contourf(xx, yy, H, cmap = cm, alpha = 0.4)
    plt.scatter(X[:,0], X[:,1], c=y, s=20, edgecolor='k')       