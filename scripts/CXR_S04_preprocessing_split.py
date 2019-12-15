#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 06:06:16 2019

@author: Alan
"""
from CXR_helpers import *

for position in positionS:
    y = np.load(path_compute + 'y_' + position + '.npy')
    labels = pd.read_pickle(path_compute + 'labels_' + position + '.pkl')
    for images_size in images_sizes:
        globals()['X_' + images_size]=np.load(path_compute + 'X_' + position + '_' + images_size +'.npy')
    
    #split training and validation, making sure that the samples of every patient are on either one or the other side
    ids = labels['Patient ID'].unique().tolist()
    np.random.shuffle(ids)
    percent_train = 0.8
    percent_val = 0.1
    n_limit_train = int(len(ids)*percent_train)
    n_limit_val = int(len(ids)*(percent_train+percent_val))
    #split IDs
    IDs={}
    IDs['train'] = ids[:n_limit_train]
    IDs['val'] = ids[n_limit_train:n_limit_val]
    IDs['test'] = ids[n_limit_val:]
    
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

    
