#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:28:10 2019

@author: Alan
"""
#load other source
from CXR_helpers import *

#load libraries and import functions
from keras import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape
from keras import regularizers
from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from keras.models import Model, model_from_json, clone_model
from keras import backend as K
from tensorflow import set_random_seed
set_random_seed(0)

#define dictionary of starting learning rates
learning_rates={'VGG16':0.00001, 'VGG19':0.00001, 'MobileNet':0.0001, 'MobileNetV2':0.0001, 'DenseNet121':0.0001, 'DenseNet169':0.0001, 'DenseNet201':0.0001, 'NASNetMobile':0.0001, 'NASNetLarge':0.0001, 'Xception':0.0001, 'InceptionV3':0.0001, 'InceptionResNetV2':0.0001}

#define dictionary of batch sizes
batch_sizes = dict.fromkeys(['VGG16', 'VGG19', 'MobileNet', 'MobileNetV2', 'NASNetMobile'], 32)
batch_sizes.update(dict.fromkeys(['DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetLarge', 'Xception', 'InceptionV3', 'InceptionResNetV2'], 16))
batch_sizes.update(dict.fromkeys(['NASNetLarge'], 8))

#Define custom loss function. source: https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
def R_squared(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
  
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def load_data(position, images_size, folds):
    yS={}
    for fold in folds:
        globals()['X_' + fold] = np.load(path_compute + 'X_' + position + '_' + images_size + '_' + fold + '.npy')
        yS[fold] = np.load(path_compute + 'y_' + position + '_' + fold + '.npy')
    return X_train, X_val, X_test, yS

def generate_base_model(model_name, lam, dropout_rate, import_weights):
    if model_name in ['VGG16', 'VGG19']:
        if model_name == 'VGG16':
            from keras.applications.vgg16 import VGG16
            base_model = VGG16(include_top=False, weights=import_weights, input_shape=(224,224,3))
        elif model_name == 'VGG19':
            from keras.applications.vgg19 import VGG19
            base_model = VGG19(include_top=False, weights=import_weights, input_shape=(224,224,3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(lam))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(lam))(x)
        x = Dropout(dropout_rate)(x) 
    elif model_name in ['MobileNet', 'MobileNetV2']:
        if model_name == 'MobileNet':
            from keras.applications.mobilenet import MobileNet
            base_model = MobileNet(include_top=False, weights=import_weights, input_shape=(224,224,3))
        elif model_name == 'MobileNetV2':
            from keras.applications.mobilenet_v2 import MobileNetV2
            base_model = MobileNetV2(include_top=False, weights=import_weights, input_shape=(224,224,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    elif model_name in ['DenseNet121', 'DenseNet169', 'DenseNet201']:
        if model_name == 'DenseNet121':
            from keras.applications.densenet import DenseNet121
            base_model = DenseNet121(include_top=True, weights=import_weights, input_shape=(224,224,3))
        elif model_name == 'DenseNet169':
            from keras.applications.densenet import DenseNet169
            base_model = DenseNet169(include_top=True, weights=import_weights, input_shape=(224,224,3))
        elif model_name == 'DenseNet201':
            from keras.applications.densenet import DenseNet201
            base_model = DenseNet201(include_top=True, weights=import_weights, input_shape=(224,224,3))            
        base_model = Model(base_model.inputs, base_model.layers[-2].output)
        x = base_model.output
    elif model_name in ['NASNetMobile', 'NASNetLarge']:
        if model_name == 'NASNetMobile':
            from keras.applications.nasnet import NASNetMobile
            base_model = NASNetMobile(include_top=True, weights=import_weights, input_shape=(224,224,3))
        elif model_name == 'NASNetLarge':
            from keras.applications.nasnet import NASNetLarge
            base_model = NASNetLarge(include_top=True, weights=import_weights, input_shape=(331,331,3))
        base_model = Model(base_model.inputs, base_model.layers[-2].output)
        x = base_model.output
    elif model_name == 'Xception':
        from keras.applications.xception import Xception
        base_model = Xception(include_top=False, weights=import_weights, input_shape=(299,299,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
    elif model_name == 'InceptionV3':
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(include_top=False, weights=import_weights, input_shape=(299,299,3))
        x = base_model.output        
        x = GlobalAveragePooling2D()(x)
    elif model_name == 'InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        base_model = InceptionResNetV2(include_top=False, weights=import_weights, input_shape=(299,299,3))
        x = base_model.output        
        x = GlobalAveragePooling2D()(x)
    return x, base_model.input

def complete_architecture(x, input_shape, lam, dropout_rate):
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(lam))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(lam))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(lam))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lam))(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs=input_shape, outputs=predictions)
    return model

def save_model_architecture(model, model_name, version, position):
    model_json = model.to_json()
    with open(path_compute + "model_" + model_name + '_' + version + '_' + position + ".json", "w") as json_file:
        json_file.write(model_json)
    print("Model's architecture for " + model_name + " was saved.")
    
def save_model_weights(model, model_name, version, position):
    model.save_weights(path_compute + "model_" + model_name + '_' + version + '_' + position + ".h5")
    print("Model's weights for "+ model_name + " were saved.")
    
def save_model_parameters(model_name, version, position, images_size, best_epoch, optimizer_name, learning_rate, batch_size, lam, dropout_rate):    
    parameters={'model':model_name, 'version': version, 'position':position, 'images_size':images_size, 'N_epochs':best_epoch, 'optimizer':optimizer_name, 'learning_rate':learning_rate, 'batch_size':batch_size, 'lambda':lam, 'dropout':dropout_rate}
    json.dump(parameters, open(path_compute + 'parameters_' + model_name + '_' + version + '_' + position,'w'))
    print("Model's parameters for "+ model_name + " were saved.")

def load_model(model_name, version, position):
    #load model's architecture
    json_file = open(path_compute + "model_" + model_name + '_' + version + '_' + position + '.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    #load weights into model
    model.load_weights(path_compute + "model_" + model_name + '_' + version + '_' + position + ".h5")
    return(model)
    
def deep_copy_model(model, optimizer_name, learning_rate):
    model_copy=clone_model(model)
    model_copy.build(model.input)
    set_learning_rate(model_copy, optimizer_name, learning_rate)
    model_copy.set_weights(model.get_weights())
    return model_copy
    
def set_learning_rate(model, optimizer_name, learning_rate):
    opt = globals()[optimizer_name](lr=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=[R_squared, root_mean_squared_error])

def initiate_metrics_training(model, X_train, X_val, yS, batch_size, folds_tune):
    R2S={}
    RMSES={}
    for fold in folds_tune:
        pred=model.predict(globals()['X_' + fold], batch_size = batch_size).squeeze()
        R2S[fold]=[r2_score(yS[fold], pred)]
        RMSES[fold]=[sqrt(mean_squared_error(yS[fold], pred))]
    return R2S, RMSES

def train_model(model, X_train, X_val, yS, batch_size, folds_tune, PREDS, R2S, RMSES):
    model.fit(X_train, yS['train'], validation_data=[X_val,yS['val']], epochs=1, batch_size=batch_size)
    for fold in folds_tune:
        PREDS[fold]=model.predict(globals()['X_' + fold].squeeze())
        R2_f=r2_score(yS[fold], PREDS[fold])
        R2S[fold].append(R2_f)
        print('R2_' + fold + ' = ' + str(R2_f))
        RMSE_f=sqrt(mean_squared_error(yS[fold], PREDS[fold]))
        RMSES[fold].append(RMSE_f) 
        print('RMSE_' + fold + ' = ' + str(RMSE_f))
            
def save_model(model, model_name, version, position, images_size, best_epoch, optimizer_name, learning_rate, batch_size, lam, dropout_rate):
    save_model_architecture(model=model, model_name=model_name, position=position, version=version)
    save_model_weights(model=model, model_name=model_name, position=position, version=version)
    save_model_parameters(model_name=model_name, version=version, position=position, images_size=images_size, best_epoch=best_epoch, optimizer_name=optimizer_name, learning_rate=learning_rate, batch_size=batch_size, lam=lam, dropout_rate=dropout_rate)
            
def plot_training(R2S, RMSES, folds_tune, model_name, version, position):
    fig, axs = plt.subplots(1, 2, sharey=False, sharex=True)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    epochs = np.array(range(len(R2S['train'])))
    #plot the R-Squared at every iteration for both training and validation
    for fold in folds_tune:
        axs[0].plot(epochs, R2S[fold])
        axs[0].legend(['Training R-Squared', 'Validation R-Squared'])
        axs[0].set_title('R-Squared = f(Epoch)')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('R-Squared')
        axs[0].set_ylim((-0.2, 1.1))
    #plot the RMSE at every iteration for both training and validation
    for fold in folds_tune:
        axs[1].plot(epochs, RMSES[fold])
        axs[1].legend(['Training RMSE', 'Validation RMSE'])
        axs[1].set_title('Root Mean Squared Error = f(Epoch)')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Root Mean Squared Error')
        axs[1].set_ylim((0, 25))
    #save figure as pdf
    fig.savefig("../figures/Tuning_" + model_name + '_' + version + '_' + position + '.pdf', bbox_inches='tight')

def generate_predictions_and_performances(model, X_train, X_val, X_test, yS, folds, model_name, version, position):
    PREDS_final={}
    R2S_final={}
    RMSES_final={}
    for fold in folds:
        PREDS_final[fold] = model.predict(globals()['X_' + fold]).squeeze()
        R2S_final[fold] = r2_score(yS[fold], PREDS_final[fold])
        RMSES_final[fold] = sqrt(mean_squared_error(yS[fold], PREDS_final[fold]))
        #save the predictions
        np.save(path_compute + 'pred_' + model_name + '_' + version + '_' + position + '_' + fold, PREDS_final[fold])
    print("R2s: " + str(R2S_final))
    print("RMSEs: " + str(RMSES_final)) 
    return PREDS_final, R2S_final, RMSES_final

def boot_performances(yS, PREDS_final, R2S_final, RMSES_final, folds, boot_iterations):
    #initiate storage of values
    R2sdS={}
    RMSEsdS={}
    for fold in folds:
        #compute performance's standard deviation using bootstrap    
        r2s = list()
        rmses = list()
        y_fold = yS[fold]
        pred_fold = PREDS_final[fold]
        n_size = len(y_fold)
        for i in range(boot_iterations):
            index_i = np.random.choice(range(n_size), size=n_size, replace=True)
            y_i = y_fold[index_i]
            pred_i = pred_fold[index_i]
            r2s.append(r2_score(y_i, pred_i))
            rmses.append(sqrt(mean_squared_error(y_i, pred_i)))    
        R2sdS[fold]=np.std(r2s)
        RMSEsdS[fold]=np.std(rmses)
        #print performance
        print("R2_" + fold + " is " + str(round(R2S_final[fold],3)) + "+-" + str(round(R2sdS[fold],3)))
        print("RMSE_" + fold + " is " + str(round(RMSES_final[fold],1)) + "+-" + str(round(RMSEsdS[fold],1)))
    return R2sdS, RMSEsdS

def save_performances(R2S_final, R2sdS, RMSES_final, RMSEsdS, folds, model_name, version, position):
    performances={}
    for fold in folds:
        performances['R2_' + fold] = str(round(R2S_final[fold], 3)) + "+-" + str(round(R2sdS[fold],3))
        performances['RMSE_' + fold]= str(round(RMSES_final[fold], 1)) + "+-" + str(round(RMSEsdS[fold],1))
    json.dump(performances, open(path_compute + 'performances_' + model_name + '_' + version + '_' + position,'w'))

def plot_performances(yS, PREDS_final, R2S_final, R2sdS, RMSES_final, RMSEsdS, folds, model_name, version, position):
    fig, axs = plt.subplots(1, len(folds), sharey=True, sharex=True)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    for k, fold in enumerate(folds):
        y_fold = yS[fold]
        pred_fold = PREDS_final[fold]
        R2_fold = R2S_final[fold]
        RMSE_fold = RMSES_final[fold]
        b_fold, m_fold = polyfit(y_fold, pred_fold, 1)
        axs[k].plot(y_fold, pred_fold, 'b+')
        axs[k].plot(y_fold, b_fold + m_fold * y_fold, 'r-')
        axs[k].set_title(dict_folds[fold] + ", N=" + str(len(y_fold)) +", R2=" + str(round(R2_fold, 3)) + "+-" + str(round(R2sdS[fold],3)) + ", RMSE=" + str(round(RMSE_fold, 1)) + "+-" + str(round(RMSEsdS[fold],1)) )
        axs[k].set_xlabel("Age")
    axs[0].set_ylabel("Predicted Age")
    #save figure
    fig.savefig("../figures/Performance_" + model_name + '_' + version + '_' + position + ".pdf", bbox_inches='tight')
    
def postprocessing(model, X_train, X_val, X_test, yS, model_name, version, position, images_size, best_epoch, optimizer_name, learning_rate, batch_size, lam, dropout_rate, R2S, RMSES, folds, folds_tune, boot_iterations):
    print('saving model architecture, weights, parameters')
    save_model(model=model, model_name=model_name, version=version, position=position, images_size=images_size, best_epoch=best_epoch, optimizer_name=optimizer_name, learning_rate=learning_rate, batch_size=batch_size, lam=lam, dropout_rate=dropout_rate)
    print('plot training of the model')
    plot_training(R2S=R2S, RMSES=RMSES, folds_tune=folds_tune, model_name=model_name, version=version, position=position)
    print('generate predictions and performances')
    PREDS_final, R2S_final, RMSES_final = generate_predictions_and_performances(model=model, X_train=X_train, X_val=X_val, X_test=X_test, yS=yS, folds=folds, model_name=model_name, version=version, position=position)
    print('bootstrapping the performances')
    R2sdS, RMSEsdS = boot_performances(yS=yS, PREDS_final=PREDS_final, R2S_final=R2S_final, RMSES_final=RMSES_final, folds=folds, boot_iterations=boot_iterations)
    print('saving the performances')
    save_performances(R2S_final=R2S_final, R2sdS=R2sdS, RMSES_final=RMSES_final, RMSEsdS=RMSEsdS, folds=folds, model_name=model_name, version=version, position=position)
    print('plotting the performances')
    plot_performances(yS=yS, PREDS_final=PREDS_final, R2S_final=R2S_final, R2sdS=R2sdS, RMSES_final=RMSES_final, RMSEsdS=RMSEsdS, folds=folds, model_name=model_name, version=version, position=position)

def tune_model(model, X_train, X_val, yS, model_name, version, position, images_size, optimizer_name, learning_rate, batch_size, lam, dropout_rate, R2S, RMSES, folds_tune, save_intermediate_model):
    PREDS={}
    max_R2_val = np.max(R2S['val'])
    best_model = deep_copy_model(model=model, optimizer_name=optimizer_name, learning_rate=learning_rate)
    best_epoch=0
    stop_shrinking = 0 #reduce learning rate until no benefit is observed
    while((learning_rate > 1e-7) & (stop_shrinking < 2)):
        stop_searching = 0
        stop_shrinking+=1
        while stop_searching < 3:
            stop_searching+=1
            print('Running epoch: ' + str(len(R2S['train'])))
            train_model(model=model, X_train=X_train, X_val=X_val, yS=yS, batch_size=batch_size, folds_tune=folds_tune, PREDS=PREDS, R2S=R2S, RMSES=RMSES)
            if R2S['val'][-1] > max_R2_val:
                stop_searching = 0
                stop_shrinking = 0
                max_R2_val = np.max(R2S['val'])
                best_epoch = len(R2S['train'])-1
                del best_model
                gc.collect()
                best_model=deep_copy_model(model=model, optimizer_name=optimizer_name, learning_rate=learning_rate)
                if save_intermediate_model:
                    save_model(model=best_model, model_name=model_name, version='temporary', position=position, images_size=images_size, best_epoch=best_epoch, optimizer_name=optimizer_name, learning_rate=learning_rate, batch_size=batch_size, lam=lam, dropout_rate=dropout_rate)
                print('The validation R2 improved.')
            print(str(stop_searching) + " epochs without improvement.")
        learning_rate/=10
        set_learning_rate(model, optimizer_name, learning_rate)
        print("Shrinking the learning rate to " + str(learning_rate))
    print('Tuning of the model completed.')
    return best_model, best_epoch

    