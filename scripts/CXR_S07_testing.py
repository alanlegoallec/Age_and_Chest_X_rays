#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:05:49 2019

@author: Alan
"""
position = 'PA'
model_name = 'VGG19'

#import libraries
from CXR_helpers import *
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

# configure bootstrap
n_iterations = 10000
#initiate storage of values
yS={}
predS={}
R2S={}
R2sdS={}
for fold in folds:
    #load target and predictions 
    yS[fold] = np.load(path_compute + 'y_' + position + '_' + fold + '.npy')
    predS[fold] = np.load(path_compute + 'pred_' + position + '_' + model_name + '_' + fold + '.npy')
    #compute performance
    R2S[fold] = r2_score(yS[fold], predS[fold])
    np.save(path_compute + 'R2_' + position + '_' + model_name + '_' + fold, R2S[fold])
    #compute performance's standard deviation using bootstrap
    stats = list()
    n_size = len(yS[fold])
    for i in range(n_iterations):
        index_i = np.random.choice(range(n_size), size=n_size, replace=True)
        stats.append(r2_score(yS[fold][index_i], predS[fold][index_i]))
    R2sdS[fold]=np.std(stats)
    np.save(path_compute + 'R2sd_' + position + '_' + model_name + '_' + fold, R2sdS[fold])
    #print performance
    print("Model: " + model_name + ". R2_" + fold + " is " + str(round(R2S[fold],3)) + "+-" + str(round(R2sdS[fold],3)))

#plot the performances on train, val and test
fig, axs = plt.subplots(1, len(folds), sharey=True, sharex=True)
fig.set_figwidth(15)
fig.set_figheight(5)
for k, fold in enumerate(folds):
    y_fold = yS[fold]
    pred_fold = predS[fold]
    R2_fold = R2S[fold]
    b_fold, m_fold = polyfit(y_fold, pred_fold, 1)
    axs[k].plot(y_fold, pred_fold, 'b+')
    axs[k].plot(y_fold, b_fold + m_fold * y_fold, 'r-')
    axs[k].set_title(fold + ", N=" + str(len(y_fold)) +", R2=" + str(round(R2_fold, 3)) + "+-" + str(round(R2sdS[fold],3)))
    axs[k].set_xlabel("Age")

axs[0].set_ylabel("Predicted Age")
#save figure
fig.savefig("../figures/Model_" + position + '_' + model_name + '_' + "performance.pdf", bbox_inches='tight')

