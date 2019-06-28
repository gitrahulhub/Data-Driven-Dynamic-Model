# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:35:51 2019

@author: rrameshbabu6
Implementation of unscented kalman filter with additive noise assumption
inputs:
    x0: initial state vector
    fx: function that predicts state given previous state and input
    hx: function that transforms state vector to measurement format
    Rn: measurement noise covariance matrix
    Rv: process noisce covariance matrix
    
"""
import numpy as np
from scipy.linalg import cholesky, sqrtm


class UKF:
    def __init__(self,x0,fx,hx,Rn,Rv,dt):
        # dimension of state vector
        self.L = len(x0)
        self.x_hat = np.array(x0)
        print(self.x_hat.shape)
        self.P = np.eye((self.L))
        
        # alpha determines spread of sigma points around mean, usually between 1e-4 and 1
        self.alpha = 0.001
        # beta incorporates prior knowledge of distribution (2 is optimal for gaussian)
        self.beta = 2 
        self.kappa = 3 - self.L
        self.lam = (self.alpha**2)*(self.L + self.kappa) - self.L
        self.gamma = (self.L + self.lam)**0.5
        
        self.fx = fx
        self.hx = hx
        self.dt = dt
        
        self.Rn = Rn
        self.Rv = Rv
        
    
    def unscented_transform(self):
        # calculate weights (1st element is m, 2nd element is c)
        self.weightsM = [self.lam/(self.L + self.lam)]
        self.weightsC = [self.lam/(self.L + self.lam)+ (1 - self.alpha**2 + self.beta)]
        for i in range(1,2*self.L + 1):
            self.weightsM = self.weightsM + [1/(2*(self.L+self.lam))]
            self.weightsC = self.weightsC + [1/(2*(self.L+self.lam))]
        
        self.sigmas = np.zeros((self.L,2*self.L+1))
        self.x_hat = self.x_hat.reshape((2,))
        self.sigmas[:,0] = self.x_hat
        U = sqrtm(self.P*(self.L+self.lam))
        for i in range(1,self.L+1):
            self.sigmas[:,i] =  self.x_hat + U[i-1,:]
            self.sigmas[:,self.L+i] =  self.x_hat - U[i-1,:]
    
    def predict(self):
        self.unscented_transform()
        
        self.sigmas_x = np.zeros((self.L,2*self.L+1))
        for i in range(2*self.L + 1):
            self.sigmas_x[:,i] = self.fx(self.sigmas[:,i],self.dt) 
            
        self.x_hat_mean = np.zeros((self.L,1))
        for i in range(2*self.L + 1):
            step1 = np.zeros((self.L,1))
            for j in range(self.L):
                step1[j] = self.weightsM[i]*self.sigmas_x[j,i]
            self.x_hat_mean += step1

        
        self.P_xx = np.zeros((self.L,self.L))
        for i in range(2*self.L + 1):
            self.P_xx += self.weightsC[i]*np.matmul(self.sigmas_x[:,i].reshape(self.x_hat_mean.shape)-self.x_hat_mean,
                                      np.transpose(self.sigmas_x[:,i].reshape(self.x_hat_mean.shape)-self.x_hat_mean))
        self.P_xx += self.Rv
        
    def update(self,y):
        self.sigmas_y = np.zeros((len(y),2*self.L+1))
        for i in range(2*self.L + 1):
            self.sigmas_y[:,i] = self.hx(self.sigmas_x[:,i]) 

        self.y_hat_mean = np.zeros((len(y),1))
        for i in range(2*self.L + 1):
            step1 = np.zeros((len(y),1))
            for j in range(len(y)):
                step1[j] = self.weightsM[i]*self.sigmas_y[j,i]
            self.y_hat_mean += step1
            
        P_yy = np.zeros((len(y),len(y)))
        P_xy = np.zeros((self.L,len(y)))
        for i in range(2*self.L + 1):
            P_yy += (self.weightsC[i]*np.matmul(self.sigmas_y[:,i].reshape(self.y_hat_mean.shape)-self.y_hat_mean,
                     np.transpose(self.sigmas_y[:,i].reshape(self.y_hat_mean.shape)-self.y_hat_mean)))
            P_xy += (self.weightsC[i]*np.matmul(self.sigmas_x[:,i].reshape(self.x_hat_mean.shape)-self.x_hat_mean,
                     np.transpose(self.sigmas_y[:,i].reshape(self.y_hat_mean.shape)-self.y_hat_mean)))
        P_yy += self.Rn
        Gain = np.matmul(P_xy,np.linalg.inv(P_yy))
        y = np.array(y)
        self.x_hat = self.x_hat_mean + np.matmul(Gain,(y - self.y_hat_mean))
        self.P = self.P_xx - np.matmul(Gain,np.matmul(P_yy,np.transpose(Gain)))
            
    
    

