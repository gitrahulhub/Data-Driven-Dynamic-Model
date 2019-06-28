# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:52:05 2019

@author: rrameshbabu6
"""
from matplotlib import pyplot as plt
from prepare_data import prepare_data
import pickle as p
import numpy as np
import csv

file = "train_neurons_xy.p"
test_data = p.load( open( file, "rb" ) )
# test_data = {'configs':configs, 'data':data, 'scaler':scaler, 'actual outputs':outputs_list, 'histories':histories_set_list}
configs = test_data['configs']
data = test_data['data']
scaler = test_data['scaler']
test_samples = test_data['test samples']
histories_set_list = test_data['histories']

actual_output = list()
rmse_set_list = list()
unscaled_predictions_set_list = list()
# preprocess output
for i in range(len(configs)):
    # create output
    config = configs[i]
    n_input = config[0]    
    print('n_input: ',n_input)
    _, test_out = prepare_data(test_samples, n_input)
    actual_output.append(test_out)
    
    # calculate rsme
    unscaled_predictions_set = list()
    rmse_set = list()
    predictions_list = data[i]
    for j in range(len(predictions_list)):
        unscaled_predictions = (scaler.inverse_transform(predictions_list[j]))
        rmse = np.sqrt(((unscaled_predictions - test_out)**2).mean(axis=0))
        unscaled_predictions_set.append(unscaled_predictions)
        rmse_set.append(rmse)
    unscaled_predictions_set_list.append(unscaled_predictions_set)
    rmse_set_list.append(rmse_set)

# average data for multiple runs
mean_rmses = list()
mean_losses = list()
mean_predictions = list()
for j in range(len(rmse_set_list)):
    # average results
    rmse_set = rmse_set_list[j]
    stacked_rmses = np.vstack(tuple(i.reshape(1,-1) for i in rmse_set))
    mean_rmse = np.mean(stacked_rmses, axis = 0)
    mean_rmses.append(mean_rmse)
    # average losses
    history_set = histories_set_list[j]
    stacked_losses = np.vstack(tuple(np.array(i['loss']).reshape(1,-1) for i in history_set))
    mean_loss = np.mean(stacked_losses, axis = 0)
    mean_losses.append(mean_loss)
    # average predictions
    predictions_set = unscaled_predictions_set_list[j]
    mean_prediction = np.mean([i for i in predictions_set], axis = 0)
    mean_predictions.append(mean_prediction)
    

config_index = 0
run_index = 0
experiment_name = 'experiments/train_neurons_xy'


for i in range(len(configs)):
    # lat and lon
    plt.figure(1)
    plt.plot(mean_predictions[config_index][:,0],mean_predictions[config_index][:,1],'r')
    plt.plot(actual_output[config_index][:,0],actual_output[config_index][:,1],'b')
    plt.title('x and y')
    plt.savefig(experiment_name + '/latlon'+str(i)+'.png')
    
    plt.figure(2)
    # lat (meters)
    ax4 = plt.subplot(2,4,1)
    ax4.plot(mean_predictions[config_index][:,0],'r')
    ax4.plot(actual_output[config_index][:,0],'b')
    ax4.title.set_text('x')
    # lon (meters)
    ax5 = plt.subplot(2,4,2)
    ax5.plot(mean_predictions[config_index][:,1],'r')
    ax5.plot(actual_output[config_index][:,1],'b')
    ax5.title.set_text('y')
    # yaw (radians)
    ax2 = plt.subplot(2,4,3)
    ax2.plot(mean_predictions[config_index][:,2],'r')
    ax2.plot(actual_output[config_index][:,2],'b')
    ax2.title.set_text('yaw')
    # v_north (m/s)?
    ax3 = plt.subplot(2,4,4)
    ax3.plot(mean_predictions[config_index][:,3],'r')
    ax3.plot(actual_output[config_index][:,3],'b')
    ax3.title.set_text('v_north')
    # v_east (m/s)?
    ax4 = plt.subplot(2,4,5)
    ax4.plot(mean_predictions[config_index][:,4],'r')
    ax4.plot(actual_output[config_index][:,4],'b')
    ax4.title.set_text('v_east')
    # angular_velocity (rad/s)
    ax5 = plt.subplot(2,4,6)
    ax5.plot(mean_predictions[config_index][:,5],'r')
    ax5.plot(actual_output[config_index][:,5],'b')
    ax5.title.set_text('angular_velocity')
    plt.savefig(experiment_name + '/others'+str(i)+'.png')
    
    # loss
    plt.figure(3)
    plt.plot(mean_losses[config_index])
    plt.title('loss')
    plt.savefig(experiment_name + '/loss'+str(i)+'.png')


print(configs)
print(mean_rmses)


with open(experiment_name + '/results.csv', 'w') as fp:
    for i in range(len(configs)):
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(list(configs[i])+list(mean_rmses[i]))



