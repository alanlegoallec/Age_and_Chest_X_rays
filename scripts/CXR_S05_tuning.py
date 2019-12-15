#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:34:01 2019
@author: Alan
VGG16
"""

#load source
from CXR_helpers_tf import *

if len(sys.argv)==1:
    sys.argv.append('NASNetLarge')
    sys.argv.append('PA')

#model version
model_name = sys.argv[1]
position = sys.argv[2]
version='v1' #default=v1. Can use 'noweights' for models with no weights imported, or temporary for models for which an older, intermediate version of the model is imported (for NASNetLarge)
images_size = str(input_size_models[model_name])
import_weights = 'imagenet' #choose between None and 'imagenet'
#compiler
optimizer_name = 'RMSprop'
learning_rate = learning_rates[model_name]
batch_size = batch_sizes[model_name]
N_epochs = 10
#regularization
lam=0.0 #regularization: weight shrinking
dropout_rate=0.05
new_model=True #choose to load trained version of model, or start from imported weights
save_intermediate_model=True #save best model at each epoch. Recommended if the job is likely to fail before completion of the tuning.

#load the data
X_train, X_val, X_test, yS = load_data(position=position, images_size=images_size, folds=folds)

#take subset to debug
#for fold in folds:
#    globals()['X_' + images_size] = globals()['X_' + images_size][:128,:,:,:]
#    yS[fold] = yS[fold][:128]

#define model
if new_model:
    x, base_model_input = generate_base_model(model_name=model_name, lam=lam, dropout_rate=dropout_rate, import_weights=import_weights)
    model = complete_architecture(x=x, input_shape=base_model_input, lam=lam, dropout_rate=dropout_rate)
    set_learning_rate(model, optimizer_name, learning_rate)
else:
    model = load_model(model_name=model_name, version=version, position=position)
    
#initiate metrics storage for training monitoring
R2S, RMSES = initiate_metrics_training(model=model, X_train=X_train, X_val=X_val, yS=yS, batch_size=batch_size, folds_tune=folds_tune)

#change the learning rate between epochs if necessary
#learning_rate*=1
#set_learning_rate(model, optimizer_name, learning_rate)

#tune the model. can stop the tuning, change some hyperparameters, and restart the tuning. the weights will be conserved.
best_model, best_epoch = tune_model(model=model, X_train=X_train, X_val=X_val, yS=yS, model_name=model_name, version=version, position=position, images_size=images_size, optimizer_name=optimizer_name, learning_rate=learning_rate, batch_size=batch_size, lam=lam, dropout_rate=dropout_rate, R2S=R2S, RMSES=RMSES, folds_tune=folds_tune, save_intermediate_model=save_intermediate_model)

#plot the training, generate the predictions, compute the performances, bootstrap the performances, save the performances, plot the performances
postprocessing(model=best_model, X_train=X_train, X_val=X_val, X_test=X_test, yS=yS, model_name=model_name, version=version, position=position, images_size=images_size, best_epoch=best_epoch, optimizer_name=optimizer_name, learning_rate=learning_rate, batch_size=batch_size, lam=lam, dropout_rate=dropout_rate, R2S=R2S, RMSES=RMSES, folds=folds, folds_tune=folds_tune, boot_iterations=boot_iterations)

print('done')