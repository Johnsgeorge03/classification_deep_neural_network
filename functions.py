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

def deriv_sigmoid(z):
    return sigmoid(z)*(1 - sigmoid(z))

def deriv_relu(z):
    return np.where(z >= 0, 1, 0)




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
    A = sigmoid(Z)
    cache = (A, W, Z, b)
    caches.append(cache)
    
    return caches
    
   
    
   
def cost_function(Al, Y_true):
    m    = Y_true.shape[1]
    
    cost = -(1/m)*np.sum(Y_true*np.log(Al)
                         + (1 - Y_true)*np.log(1 - Al))
    cost = np.squeeze(cost)
    
    return cost




def backward_prop(caches, Y):
    grads   = {}
    L       = len(caches) + 1
    (AL, WL, ZL, bL) = caches[-1]
    m       = AL.shape[1]
    Y       = Y.reshape(AL.shape)
    
    dAL     = - (Y/AL - (1 - Y)/(1 - AL))
    
    dZL     =  dAL * deriv_sigmoid(ZL)
    
    dWL     = (1/m)*dZL@(caches[-2][0]).T
    
    dbL     = (1/m)*np.sum(dZL, axis = 1, keepdims = True)
    
    dA_prev = WL.T@dZL
    
    grads['dA' + str(L - 2)] = dA_prev
    grads['dW' + str(L - 1)] = dWL
    grads['db' + str(L - 1)] = dbL
    
    for l in reversed(range(L-2)):
        (Al, Wl, Zl, bl) = caches[l+1]
        dZl     = dA_prev * deriv_relu(Zl)
        dWl     = (1/m)*dZl@(caches[l][0]).T
        dbl     = (1/m)*np.sum(dZl, axis = 1, keepdims = True)
        dA_prev = Wl.T@dZl
        
        grads['dA' + str(l)]     = dA_prev
        grads['dW' + str(l + 1)] = dWl
        grads['db' + str(l + 1)] = dbl
        
    
    return grads




def update_parameteres(params, grads, learning_rate):
    
    parameters = params.copy()
    L = len(parameters) + 1
    
    for l in range(1, L):
        parameters['W' + str(l)] = params['W' + str(l)] - learning_rate*grads['dW' + str(l)]
        parameters['b' + str(l)] = params['b' + str(l)] - learning_rate*grads['db' + str(l)]
    
    
    return parameters




def L_layer_model(X, Y, layer_dims, learning_rate = 0.075, num_iterations = 1000, print_cost=False):
    np.random.seed(1)
    costs = []
    
    parameters  = initialize_parameters(layer_dims)
    
    for i in range(num_iterations):
        caches = forward_propagation(X, parameters)
        
        cost   = cost_function(caches[-1][0], Y)
        
        grads  = backward_prop(caches, Y)
        
        parameters = update_parameteres(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    