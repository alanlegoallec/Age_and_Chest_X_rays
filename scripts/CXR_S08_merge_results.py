#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:46:28 2019

@author: Alan
"""

#Merge results

#load source
from CXR_helpers import *

version='v1'

models_names = ['VGG16', 'VGG19', 'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge', 'Xception', 'InceptionV3', 'InceptionResNetV2']

for position in positionS:
    print("Summary results for: " + dict_positions[position])
    #initiate columns
    for subset in ['train', 'val']:
        for metric in ['R2', 'RMSE']:
            globals()[metric + '_' + subset] = []
            
    #fill columns
    for model_name in models_names:
        performance_model = json.load(open(path_compute + 'performances_' + model_name + '_' + version + '_' + position))
        for subset in ['train', 'val']:
            for metric in ['R2', 'RMSE']:
                globals()[metric + '_' + subset].append(performance_model[metric + '_' + subset])
    
    #Merge and convert to a dataframe
    Performances = {'Architecture':models_names}
    for subset in ['train', 'val']:
        for metric in ['R2', 'RMSE']:
            Performances[metric + '_' + dict_folds[subset]] = globals()[metric + '_' + subset]
    Performances = pd.DataFrame(Performances)
    Performances.set_index('Architecture')
    
    #save dataframe
    Performances.to_csv(path_store + 'Performances_' + position + '.csv', index=False, sep='\t')