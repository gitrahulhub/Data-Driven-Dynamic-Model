# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:53:46 2019

@author: rrameshbabu6
"""
from gridsearch import GridSearch
from prepare_data import scale_data
from prepare_sensor_data import load_data
import pickle as p

# load dataset
roboBoat_gridsearch = GridSearch(time_series = True)

train_file = 'train.json'
test_file = 'test.json'

train_samples = load_data(train_file) 
test_samples = load_data(test_file)

train_samples = train_samples[0:15000,:]
test_samples = test_samples[0:5000,:]

# transform the scale of the data
scaler, train_scaled,test_scaled = scale_data(train_samples, test_samples)

data, histories_set_list = roboBoat_gridsearch.grid_search(train_scaled, test_scaled)
    
# save data to pickle
test_data = {'configs':roboBoat_gridsearch.configs, 'data':data, 'scaler':scaler, 'test samples':test_samples, 'histories':histories_set_list}
p.dump( test_data, open( "train_epochs_xy.p", "wb" ) )