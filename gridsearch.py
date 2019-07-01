# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:21:43 2019

@author: rrameshbabu6

this class is to tune the hyperparameters of a MLP using the grid-search method

"""
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import itertools
import time
from prepare_data import make_data_supervised

class GridSearch:
    def __init__(self,n_input = [1], n_nodes = [100], n_epochs = [100], 
                 n_batch = [10], n_hidden_layers = [0],
                 dropout = [0], activation = ['relu'], time_series = False):
        self.configs = self.model_configs(n_input, n_nodes, n_epochs, n_batch, 
                                     n_hidden_layers, dropout, activation)
        self.time_series = time_series
            
    # fit a model to training data
    def model_fit(self,scaled_train_in,scaled_train_out, config):
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
        model.add(Dense(scaled_train_out.shape[1]))
        model.compile(loss='mse', optimizer='adam')
        
        # fit model
        history = model.fit(scaled_train_in, scaled_train_out, epochs=n_epochs, batch_size=n_batch, verbose=0)
        return model, history
    
    def validate(self,scaled_train_in, scaled_train_out, scaled_test_in,
                                scaled_test_out, cfg):
        # unpack config
        n_input, _, _, _, _, _, _ = cfg 
        
    	# fit model
        model, history = self.model_fit(scaled_train_in, scaled_train_out, cfg)
    	# make predictions
        predictions = model.predict(scaled_test_in.reshape(-1,scaled_train_in.shape[1]*n_input)) 
        model.reset_states()
        return predictions, history.history
    
    # create a list of configs to try
    def model_configs(self, n_input, n_nodes, n_epochs, n_batch, 
                      n_hidden_layers, dropout, activation):
        configs = list(itertools.product(n_input,n_nodes,n_epochs,n_batch,
                                         n_hidden_layers,dropout,activation))
        return configs
    
    def repeat_evaluate(self,train_scaled_in, 
                        train_scaled_out, test_scaled_in, test_scaled_out, config, n_repeats=1):
        n_input, _, _, _, _, _, _ = config 
        
        # if this is a time_series problem, prepare data accordingly
        if self.time_series:
            train_scaled_in, train_scaled_out = make_data_supervised(train_scaled_in, n_input)
            test_scaled_in, test_scaled_out = make_data_supervised(test_scaled_in, n_input)
            
        print('number of observations: ', config[0])
        print('number of nodes: ', config[1])
        print('number of epochs: ', config[2])
        print('number of hidden layers: ', config[4])
        print('number of batches: ', config[3])
        
    	# fit and evaluate the model n times
        predictions_list = list()
        history_list = list()
        
        for i in range(n_repeats):
            predictions, history = self.validate(train_scaled_in, 
                        train_scaled_out, test_scaled_in, test_scaled_out, config)
            
            predictions_list.append(predictions)
            history_list.append(history)
            
        return predictions_list, history_list
    
    # grid search configs
    def grid_search(self,train_scaled_in, 
                         test_scaled_in, test_scaled_out = None, train_scaled_out = None):
    	# evaluate configs
        data = list()
        histories_list = list()
        for cfg in self.configs:
            start = time.time()
            predictions, histories = self.repeat_evaluate(train_scaled_in, 
                        train_scaled_out, test_scaled_in, test_scaled_out, cfg) 
            data.append(predictions)
            histories_list.append(histories)
            elapsed = time.time() - start
            print('training time: ',elapsed)
        return data, histories_list
    