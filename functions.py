#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:59:32 2022

@author: john
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def relu(z):
    return z * (z > 0)

def initialize_parameters(layer_dims):
    """
    

    Parameters
    ----------
    layer_dims : TYPE - numpy array, size = L
        DESCRIPTION.  - contains the no. of nodes in each layer.
                        Also includes the input and output layers.
    Returns
    -------
    parameters : TYPE - dictionary of arrays
        DESCRIPTION.  - contains the parameters like W1, b1, .. WL, bL
                        Wi - weights associated to i-th layer
                        bi - bias vector associated to layer i

    """
    L = len(layer_dims)
    parameters = {}
    for l in range(1, L):
        #one-based indexing i.e., first hidden layer is W1
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters



def forward_propagation(X, parameters):
    """
    Implementation the forward propagation.
    All hidden layer activations are relu 
    and output layer activation is sigmoid.
    
    Funtion associated to all layer is linear
    i.e., Z[l] = W[l]*A[l-1] + b[l] - [l] indicates layer
    Parameters
    ----------
    X : TYPE - numpy array of size (n, m)
                n - no. of variables in x
                m - no. of training samples
        DESCRIPTION.
            - Contains the input variable examples
            
    parameters : TYPE - dictionary
        DESCRIPTION.
            - Contains the weights and biases associated
                to each layer                    

    Returns
    -------
    caches : TYPE - list of tuples
        DESCRIPTION.
            - each tuple contain the the activation of each layer,
            linear part, biases, and weights (A, W, Z, b)
        
    """
    caches = []
    A = X
    L = len(parameters) + 1
    for l in range(1, L - 1):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = W@A + b 
        A = relu(Z)
        cache = (A, W, Z, b)
        caches.append(cache)
    
    W = parameters['W' + str(L - 1)]
    b = parameters['b' + str(L - 1)]
    Z = W@A + b 
    A = relu(Z)
    cache = (A, W, Z, b)
    caches.append(cache)
    
    return caches
    
    
def cost_function(Y_est, Y_true):
    m = Y_true.shape[1]
    
    cost = -(1/m)*np.sum(np.log(Y_est)*Y_true
                         + (1 - Y_true)*np.log(1 - Y_est))
    cost = np.squeeze(cost)
    
    return cost


def backward_prop()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    