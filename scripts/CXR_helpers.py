#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:28:10 2019

@author: Alan
"""

#load libraries and import functions
import os
import tarfile
import shutil
import sys
import json
import numpy as np
import pandas as pd
import pickle
import gc
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import random
random.seed(0)

if '/Users/Alan/' in os.getcwd():
    os.chdir('/Users/Alan/Desktop/Aging/CXR/scripts/')
    path_store = '../data/'
    path_compute = '../data/'
else:
    os.chdir('/n/groups/patel/Alan/Aging/CXR/scripts/')
    path_store = '../data/'
    path_compute = '/n/scratch2/al311/Aging/CXR/data/'

folds = ['train', 'val', 'test']
folds_tune = ['train', 'val']
images_sizes = ['224', '299', '331']
positions = ['PA', 'AP']
positionS = ['PA', 'AP', 'BOTH']
boot_iterations=10000

models_names = ['VGG16', 'VGG19', 'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge', 'Xception', 'InceptionV3', 'InceptionResNetV2']
#define dictionary to resize the images to the right size depending on the model
input_size_models = dict.fromkeys(['VGG16', 'VGG19', 'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile'], 224)
input_size_models.update(dict.fromkeys(['Xception', 'InceptionV3', 'InceptionResNetV2'], 299))
input_size_models.update(dict.fromkeys(['NASNetLarge'], 331))


#define dictionaries to format the text
dict_folds={'train':'Training', 'val':'Validation', 'test':'Testing'}
dict_positions={'PA':'Posteroanterior', 'AP':'Anteroposterior', 'BOTH':'Either View'}

