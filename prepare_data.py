# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:45:54 2019

@author: rrameshbabu6
"""
import numpy as np
import json
import scipy.signal
import math
from sklearn.preprocessing import MinMaxScaler

def convert_gps_to_xy(lat_data, lon_data):
    lat0 = lat_data[0]
    lon0 = lon_data[0]
    earth_radius = 6.371 * (10**6) 
    x = (lon_data - lon0)*(math.pi/180)*math.cos(lat0)*earth_radius
    y = (lat_data - lat0)*(math.pi/180)*earth_radius
    return x,y

def convert_time_to_dt(raw_data):
    print('Converting time to dt')

    for i in range(len(raw_data)-1):
        raw_data[i, 8] = raw_data[i+1, 8] - raw_data[i, 8]
    return raw_data

#Input format is 'hh:mm:ss.ddd'
#Output format is %.f
def convert_time(timeString):
    t = float(timeString[:2]) * 60 *60
    t += float(timeString[3:5]) * 60
    t += float(timeString[6:])
    return t

def interpolate_gps(raw_data):
    print('Interpolating GPS states')

    for i in range(len(raw_data)):
        if np.isnan(raw_data[i, 0]) and i != 0 and i != len(raw_data)-2:
            previous = None
            next = None

            p = 1   #Get the index of the previous gps packet
            keep_looking = True
            while i-p >= 0 and keep_looking:
                if not np.isnan(raw_data[i-p, 0]):
                    previous = i-p
                    keep_looking = False
                p += 1
            p = 1   #Get the index of the next gps packet
            keep_looking = True
            while i+p < len(raw_data)-2 and keep_looking:
                if not np.isnan(raw_data[i+p, 0]):
                    next = i+p
                    keep_looking = False
                p += 1

            #Interpolate the gps data
            if previous != None and next != None:
                if raw_data[next, 8] - raw_data[previous, 8] > 0.001:
                    slopes = (raw_data[next, 0:2] - raw_data[previous, 0:2]) / (raw_data[next, 8] - raw_data[previous, 8])
                    raw_data[i, 0:2] = slopes * (raw_data[i, 8] - raw_data[previous, 8]) + raw_data[previous, 0:2].copy()

                    slopes = (raw_data[next, 4:6] - raw_data[previous, 4:6]) / (raw_data[next, 8] - raw_data[previous, 8])
                    raw_data[i, 4:6] = slopes * (raw_data[i, 8] - raw_data[previous, 8]) + raw_data[previous, 4:6].copy()
                else:
                    raw_data[i, 0:2] = (raw_data[next, 0:2].copy() + raw_data[previous, 0:2].copy()) / 2.
                    raw_data[i, 4:6] = (raw_data[next, 4:6].copy() + raw_data[previous, 4:6].copy()) / 2.

    return raw_data

def interpolate_imu(raw_data):
    print('Interpolating IMU states')

    for i in range(len(raw_data)):
        if np.isnan(raw_data[i, 2]) and i != 0 and i != len(raw_data)-2:
            previous = None
            next = None

            p = 1   #Get the index of the previous imu packet
            keep_looking = True
            while i-p >= 0 and keep_looking:
                if not np.isnan(raw_data[i-p, 2]):
                    previous = i-p
                    keep_looking = False
                p += 1
            p = 1   #Get the index of the next imu packet
            keep_looking = True
            while i+p < len(raw_data)-2 and keep_looking:
                if not np.isnan(raw_data[i+p, 2]):
                    next = i+p
                    keep_looking = False
                p += 1

            #Interpolate the imu states
            if previous != None and next != None:
                if raw_data[next, 8] - raw_data[previous, 8] > 0.001:
                    slopes = (raw_data[next, 2:4] - raw_data[previous, 2:4]) / (raw_data[next, 8] - raw_data[previous, 8])
                    raw_data[i, 2:4] = slopes * (raw_data[i, 8] - raw_data[previous, 8]) + raw_data[previous, 2:4].copy()
                else:
                    raw_data[i, 2:4] = (raw_data[next, 2:4].copy() + raw_data[previous, 2:4].copy()) / 2.

    return raw_data

def transform_measurements(raw_data):
    inputs = []

    #Transform from measurements into input states
    for i in range(len(raw_data)):
        inputs.append([None]*9)

        if not np.isnan(raw_data[i, 0]):    #lat = lat
            inputs[i][0] = raw_data[i, 0]
        if not np.isnan(raw_data[i, 1]):    #lon = lon
            inputs[i][1] = raw_data[i, 1]
        if not np.isnan(raw_data[i, 2]):    #theta = atan2(magY, magX)
            inputs[i][2] = math.atan2(raw_data[i, 3], raw_data[i, 2])
        if not np.isnan(raw_data[i, 4]):    #omega = gyroZ
            inputs[i][3] = raw_data[i, 4]
        if not np.isnan(raw_data[i, 5]):    #u = northVelocity
            inputs[i][4] = raw_data[i, 5]
        if not np.isnan(raw_data[i, 6]):    #v = eastVelocity
            inputs[i][5] = raw_data[i, 6]
        if not np.isnan(raw_data[i, 7]):    #mL = motor_l
            inputs[i][6] = raw_data[i, 7]
        if not np.isnan(raw_data[i, 8]):    #mR = motor_r
            inputs[i][7] = raw_data[i, 8]
        if not np.isnan(raw_data[i, 9]):    #time = time
            inputs[i][8] = raw_data[i, 9]

    return np.array(inputs, dtype=np.float)

def remove_nans(raw_data):
    print('Removing NaNs')
    return raw_data[~np.isnan(raw_data).any(axis=1)]


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


def load_data(FILENAME):
    with open(FILENAME, 'r') as fi:
        data = json.load(fi)

    num_cases = len(data)
    print('Total cases in JSON dict:', num_cases)

    temp = []

    i = 0 #Data index
    for d in data:
        temp.append([None]*10)

        #Get IMU data
        if 'scaledMagX' in d['ObjectData']:
            temp[i][2] = d['ObjectData']['scaledMagX']
            temp[i][3] = d['ObjectData']['scaledMagY']
            temp[i][4] = d['ObjectData']['scaledGyroZ']

        if 'latitude' in d['ObjectData']:
            temp[i][0] = d['ObjectData']['latitude']
            temp[i][1] = d['ObjectData']['longitude']
            temp[i][5] = d['ObjectData']['northVelocity']
            temp[i][6] = d['ObjectData']['eastVelocity']

        #Get the motor commands
        if 'Generic Boat Left' in d['ObjectData']:
            temp[i][7] = d['ObjectData']['Generic Boat Left']
            temp[i][8] = d['ObjectData']['Generic Boat Right']

        #Get the delta_time
        temp[i][9] = convert_time(d['TimestampCPU'][11:])

        i += 1
    #Clean the Data
    inputs = transform_measurements(np.array(temp, dtype=np.float))
    inputs = interpolate_imu(inputs.copy())
    inputs = interpolate_gps(inputs.copy())
    inputs = convert_time_to_dt(inputs.copy())
    inputs = remove_nans(inputs.copy())
    samples = inputs
    # convert gps to x,y
    lat_data = samples[:,0]
    lon_data = samples[:,1]
    x,y = convert_gps_to_xy(lat_data,lon_data)
    samples[:,0] = x
    samples[:,1] = y
    
    return samples

def prepare_data(samples, n_inputs):
    #Smooth eveything but the motor commands and dt
    samples[:, :6] = np.transpose(scipy.signal.savgol_filter(np.transpose(samples[:, :6].copy()), 51, 3))
    
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
    scaler = scaler.fit(np.vstack((train_samples[:,0:6],test_samples[:,0:6])))

	# transform train_samples
    train_motor_dts = train_samples[:,6:9]
    train_scaled = scaler.transform(train_samples[:,0:6])
    train_scaled = np.hstack((train_scaled,train_motor_dts))
    
	# transform test_samples
    test_motor_dts = test_samples[:,6:9]
    test_scaled = scaler.transform(test_samples[:,0:6])
    test_scaled = np.hstack((test_scaled,test_motor_dts))

    return scaler, train_scaled, test_scaled