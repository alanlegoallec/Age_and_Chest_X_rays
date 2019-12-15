#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 03:42:07 2019

@author: Alan
"""
from CXR_helpers import *

#concatenate preprocessed segments
Xs={}
for images_size in images_sizes:
    Xs[images_size]=[]
Ys = []
LABELS = []
for i in range(1,13):
    i = "{0:0=3}".format(i)
    print(i)
    Ys.append(np.load(path_compute + 'y_' + i + '.npy'))
    LABELS.append(pd.read_pickle(path_compute + 'labels_' + i + '.pkl'))
    for images_size in images_sizes:
        Xs[images_size].append(np.load(path_compute + 'X_' + images_size + '_' + i + '.npy'))
X={}
for images_size in images_sizes:
    X[images_size]=np.concatenate(Xs[images_size])
y = np.concatenate(Ys)
labels = pd.concat(LABELS)

#split into PA and AP
y_positions={}
labels_positions={}
X_positions={}
#both positions
y_positions['BOTH']=y
labels_positions['BOTH']=labels
X_positions['BOTH']={}
for images_size in images_sizes:
    X_positions['BOTH'][images_size] = X[images_size]
#PA and AP
for position in positions:
    index = np.where(labels['View Position'] == position)[0]
    y_positions[position] = y[index]
    labels_positions[position] = labels.loc[labels['View Position'] == position,:]
    X_positions[position]={}
    for images_size in images_sizes:
        X_positions[position][images_size] = X[images_size][index,:,:,:]


#display dimensions and save files for all, PA and AP data
for position in positionS:
    #display dimensions
    print("y_" + position + "'s shape is " + str(y_positions[position].shape))
    print("labels_" + position + "'s shape is " + str(labels_positions[position].shape))
    for images_size in images_sizes:
        print("X_" + position + '_' + images_size + "'s shape is " + str(X_positions[position][images_size].shape))
    #save files
    np.save(path_compute + 'y_' + position, y_positions[position])
    labels_positions[position].to_pickle(path_compute + 'labels_' + position + '.pkl')
    for images_size in images_sizes:
        np.save(path_compute + 'X_' + position + '_' + images_size, X_positions[position][images_size])


