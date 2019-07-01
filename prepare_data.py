# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:23:30 2019

@author: rrameshbabu6
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def make_data_supervised(samples, n_inputs):    
    #Make the outputs by shifting so that outputs[k, :] = inputs[k+1, :]
    inputs, outputs  = split_sequences(samples, n_inputs)
    outputs = outputs[:, :6] # ignore dt and motor commands
    
    # flatten data
    n_input = inputs.shape[1] * inputs.shape[2]
    inputs = inputs.reshape((inputs.shape[0], n_input))
    
    return inputs, outputs

def scale_data(train_samples,test_samples):
    # fit scaler to all features except dts and motor cmds
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(np.vstack((train_samples,test_samples)))
	# transform train_samples
    train_scaled = scaler.transform(train_samples)
    
	# transform test_samples
    test_scaled = scaler.transform(test_samples)

    return scaler, train_scaled, test_scaled