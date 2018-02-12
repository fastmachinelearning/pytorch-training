#!/usr/bin/env python
# encoding: utf-8

# File        : models.py
# Author      : Zhenbin Wu
# Contact     : zhenbin.wu@gmail.com
# Date        : 2018 Feb 08
#
# Description : 

import torch
import torch.nn as nn

def two_layer_model(Inputs, nclasses, l1Reg=0):
    """
    One hidden layer model
    """
    # x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', 
              # name='fc1_relu', W_regularizer=l1(l1Reg))(Inputs)
    # predictions = Dense(nclasses, activation='sigmoid', kernel_initializer='lecun_uniform', 
                        # name = 'output_sigmoid', W_regularizer=l1(l1Reg))(x)
    # model = Model(inputs=Inputs, outputs=predictions)
    model = nn.Sequential(
        torch.nn.Linear(Inputs, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, nclasses),
        # torch.nn.ReLU(),
        torch.nn.Sigmoid(),
    )
    return model

