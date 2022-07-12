#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:54:25 2022

@author: john
"""
from functions import *
import numpy as np
import matplotlib.pyplot as plt

layer_dims = np.array([2, 3, 1])
x1 = np.linspace(-1.7, 1.7, 1000)
x2 = np.linspace(-1.7, 1.7, 1000)

x1test = np.linspace(-1.7, 1.7, 1000)
x2test = np.linspace(-1.7, 1.7, 1000)

np.random.shuffle(x1test)
np.random.shuffle(x2test)
np.random.shuffle(x1)
np.random.shuffle(x2)

X_train, Y_train, _ , _ = create_train_test_set(x1, x2, 1)

X_test, Y_test, _, _ = create_train_test_set(x1test, x2test, 1)


parameters, costs = L_layer_model(X_train, Y_train, layer_dims, 
                     num_iterations = 10000, learning_rate = 1.2,  print_cost = True)



X, Y = np.meshgrid(X_train[0], X_train[1])
Y_train = Y_train*(0*X + 1)
Z = curves(X, Y, Shape = "heart")

fig = plt.figure()
plt.imshow((Z < 1e-15),extent=[-2,2,-2,2])
plt.show()