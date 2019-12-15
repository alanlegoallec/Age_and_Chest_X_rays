#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 06:07:20 2019

@author: Alan
"""
position = 'PA'
model_name = 'VGG19'

#import libraries
from CXR_helpers import *
from keras.models import model_from_json

#load model
json_file = open(path_compute + 'model_' + model_name + '.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
#load weights into model
model.load_weights(path_compute + 'model_' + model_name + ".h5")
#initiate storage of values
XS={}
yS={}
predS={}
for fold in folds:
    #load data
    XS[fold] = np.load(path_compute + 'X_' + position + '_' + fold + '.npy')
    yS[fold] = np.load(path_compute + 'y_' + position + '_' + fold + '.npy')
    #generate predictions
    predS[fold] = model.predict(XS[fold]).squeeze()
    np.save(path_compute + 'pred_' + position + '_' + model_name + '_' + fold, predS[fold])

print('done')

