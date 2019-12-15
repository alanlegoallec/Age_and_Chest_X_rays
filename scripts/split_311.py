#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:49:55 2019

@author: Alan
"""

#split images 331

from CXR_helpers import *

images_sizes=['224', '299', '331']

for position in positionS:
    y = np.load(path_compute + 'y_' + position + '.npy')
    labels = pd.read_pickle(path_compute + 'labels_' + position + '.pkl')
    for images_size in images_sizes:
        globals()['X_' + images_size]=np.load(path_compute + 'X_' + position + '_' + images_size +'.npy')
    
    #split IDs
    IDs={}
    for fold in folds:
        IDs[fold]=pd.read_pickle(path_compute + 'labels_' + position + '_' + fold + '.pkl')['Patient ID'].unique().tolist()
    
    #split, print the dimension and save the train, val and test sets
    for fold in folds:
        #split the data between folds
        indices_fold = np.where(np.isin(labels['Patient ID'], IDs[fold]))[0]
        y_fold = y[indices_fold]
        labels_fold = labels.iloc[indices_fold]
        for images_size in images_sizes:
            globals()['X_fold_' + images_size]=globals()['X_' + images_size][indices_fold,:,:,:]
            
        #print the dimension of the different folds
        print("y_" + fold + "'s shape is " + str(y_fold.shape))
        print("labels_" + fold + "'s shape is " + str(labels_fold.shape))
        for images_size in images_sizes:
            print("X_" + fold + '_' + images_size + "'s shape is " + str(globals()['X_fold_' + images_size].shape))
        #save the data
        np.save(path_compute + 'y_' + position + '_' + fold, y_fold)
        labels_fold.to_pickle(path_compute + 'labels_' + position + '_' + fold + '.pkl')
        for images_size in images_sizes:
            np.save(path_compute + 'X_' + position + '_' + images_size + '_' + fold, globals()['X_fold_' + images_size])

    
