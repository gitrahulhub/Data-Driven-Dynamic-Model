# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:21:43 2019

@author: rrameshbabu6

this class is to tune the hyperparameters of a MLP using the grid-search method

"""
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope
from keras.models import model_from_json
import itertools
# import time
from prepare_data import make_data_supervised
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

class GridSearch:
    def __init__(self,n_input = [1], n_nodes = [100], n_epochs = [100], 
                 n_batch = [10], n_hidden_layers = [0],
                 dropout = [1], activation = ['relu'], time_series = False):
        self.n_input = n_input
        self.n_nodes = n_nodes
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        self.activation = activation
        # generate initial configs
        self.generate_configs()
        # other variables
        self.time_series = time_series
        self.models = list()
            
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
    
    def validate(self,train_scaled_in, train_scaled_out, test_scaled_in,
                                test_scaled_out, cfg):
        # unpack config
        n_input, _, _, _, _, _, _ = cfg 
    	# fit model
        model, history = self.model_fit(train_scaled_in, train_scaled_out, cfg)
    	# make predictions
        predictions = None
        if test_scaled_in != None:
            predictions = model.predict(test_scaled_in.reshape(-1,train_scaled_in.shape[1]*n_input)) 
        model.reset_states()
        return predictions, history.history, model
    
    # create a list of configs to try
    def generate_configs(self):
        self.configs = list(itertools.product(self.n_input,self.n_nodes,self.n_epochs,self.n_batch,
                                         self.n_hidden_layers,self.dropout,self.activation))

    
    def repeat_evaluate(self,train_scaled_in, 
                        train_scaled_out, test_scaled_in, test_scaled_out, config, n_repeats=1):
        n_input, _, _, _, _, _, _ = config 
        
        # if this is a time_series problem, prepare data accordingly
        if self.time_series and test_scaled_in != None:
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
            predictions, history, model = self.validate(train_scaled_in, 
                        train_scaled_out, test_scaled_in, test_scaled_out, config)
            if i == 0:
                self.models.append(model)
            
            predictions_list.append(predictions)
            history_list.append(history)
            
        return predictions_list, history_list
    
    # grid search configs
    def grid_search(self,train_scaled_in, 
                         train_scaled_out = None, test_scaled_in = None, test_scaled_out = None):
        # clear models list
        self.models.clear()
    	# evaluate configs
        data = list()
        histories_list = list()
        for cfg in self.configs:
            # start = time.time()
            predictions, histories = self.repeat_evaluate(train_scaled_in, 
                        train_scaled_out, test_scaled_in, test_scaled_out, cfg) 
            data.append(predictions)
            histories_list.append(histories)
            # elapsed = time.time() - start
            # print('training time: ',elapsed)
        return data, histories_list
    
    def dumpModel(self, model, name):
        # serialize model to JSON
        modelJson = model.to_json()
        with open(name + ".json", "w") as json_file:
            json_file.write(modelJson)
        # serialize weights to HDF5
        model.save_weights(name + ".h5")
        print("Saved model to disk")

    def loadModel(self, name):
        # load json and create model
        jsonFile = open(name + '.json', 'r')
        loadedModelJson = jsonFile.read()
        jsonFile.close()
        
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            loadedModel = model_from_json(loadedModelJson)
    
        # load weights into new model
        loadedModel.load_weights(name + ".h5")
        print("Loaded model from disk")     
        loadedModel.compile(loss='mse', optimizer='adam')
        return loadedModel