# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:21:43 2019

@author: rrameshbabu6

this script is to tune the hyperparameters of a MLP using the grid-search method
that learns how to predict roboboat dynamics

"""
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np
import pickle as p
import itertools
import time
from prepare_data import prepare_data, load_data, scale_data

# fit a model to training data
def model_fit(scaled_train_in,scaled_train_out, config):
    # unpack config
    n_input, n_nodes, n_epochs, n_batch, n_hidden_layers, dropout, act  = config
    
	# define model
    model = Sequential()
    # input layer
    model.add(Dense(n_nodes, activation=act, input_dim=scaled_train_in.shape[1]))
    # hidden layers
    for i in range(n_hidden_layers):
        model.add(Dense(n_nodes, activation=act))
        model.add(Dropout(0.5))
    # output layer
    model.add(Dense(6))
    model.compile(loss='mse', optimizer='adam')
    
    # fit model
    history = model.fit(scaled_train_in, scaled_train_out, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model, history

def validate(scaled_train_in, scaled_train_out, scaled_test_in,
                            scaled_test_out, cfg):
    # unpack config
    n_input, _, _, _, _, _, _ = cfg 
    
	# fit model
    model, history = model_fit(scaled_train_in, scaled_train_out, cfg)
	# make predictions
    predictions = model.predict(scaled_test_in.reshape(-1,9*n_input)) 
    model.reset_states()
    return predictions, history.history

# create a list of configs to try
def model_configs():
    n_input = [3]
    n_nodes = [2000]
    n_epochs = [500, 1000, 2000, 4000]
    n_batch = [100]
    n_hidden_layers = [1]
    dropout = [0] # probablility of dropout
    activation = ['relu']

    configs = list(itertools.product(n_input,n_nodes,n_epochs,n_batch,
                                     n_hidden_layers,dropout,activation))
    
    return configs

# score a model, return None on failure
def repeat_evaluate(train_scaled, test_scaled, config, n_repeats=1):
    n_input, _, _, _, _, _, _ = config 
    train_scaled_in, train_scaled_out = prepare_data(train_scaled, n_input)
    test_scaled_in, test_scaled_out = prepare_data(test_scaled, n_input)
    print('number of observations: ', config[0])
    print('number of nodes: ', config[1])
    print('number of epochs: ', config[2])
    print('number of hidden layers: ', config[4])
    print('number of batches: ', config[3])
    
	# fit and evaluate the model n times
    predictions_list = list()
    history_list = list()
    
    for i in range(n_repeats):
        predictions, history = validate(train_scaled_in, 
                    train_scaled_out, test_scaled_in, test_scaled_out, config)
        
        predictions_list.append(predictions)
        history_list.append(history)
        
    return predictions_list, history_list

# grid search configs
def grid_search(train_scaled, test_scaled, cfg_list):
    
	# evaluate configs
    data = list()
    histories_list = list()
    for cfg in cfg_list:
        start = time.time()
        predictions, histories = repeat_evaluate(train_scaled, test_scaled, cfg) 
        data.append(predictions)
        histories_list.append(histories)
        elapsed = time.time() - start
        print('training time: ',elapsed)
    return data, histories_list

# load dataset
train_file = 'train.json'
test_file = 'test.json'

configs = model_configs()

train_samples = load_data(train_file) 
test_samples = load_data(test_file)

train_samples = train_samples[0:15000,:]
test_samples = test_samples[0:5000,:]

# transform the scale of the data
scaler, train_scaled,test_scaled = scale_data(train_samples, test_samples)

data, histories_set_list = grid_search(train_scaled, test_scaled, configs)
    
# save data to pickle
test_data = {'configs':configs, 'data':data, 'scaler':scaler, 'test samples':test_samples, 'histories':histories_set_list}
p.dump( test_data, open( "train_neurons_xy.p", "wb" ) )


