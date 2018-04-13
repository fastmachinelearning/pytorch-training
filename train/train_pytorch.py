#!/usr/bin/env python3
# encoding: utf-8

# File        : train_pytorch.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2018 Apr 05
#
# Description :

import sys
import os
import yaml
import models
import torch

from sklearn.metrics import roc_curve, auc
from optparse import OptionParser
from torch import nn
from torch.autograd import Variable
from Data_loader import get_features, parse_config
from Data_loader import tensor_toVar, numpy_toVar, TorchDataset

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-i','--input'   ,action='store',type='string',dest='inputFile'   ,default='../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z', help='input file')
    parser.add_option('-t','--tree'   ,action='store',type='string',dest='tree'   ,default='t_allpar_new', help='tree name')
    parser.add_option('-g','--disableGPU'   ,action='store_false', dest='noCUDA'   ,default=False, help='Dont run with CUDA')
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
    parser.add_option('-c','--config'   ,action='store',type='string', dest='config', default='train_config_threelayer.yml', help='configuration file')
    # parser.add_option('-c','--config'   ,action='store',type='string', dest='config', default='train_config_twolayer.yml', help='configuration file')
    (options,args) = parser.parse_args()

    yamlConfig = parse_config(options.config)

    if os.path.isdir(options.outputDir):
        os.removedirs(options.outputDir)
    else:
        os.mkdir(options.outputDir)

    ## Getting the train and test sample
    X_train_val, X_test_val, Y_train_val, Y_test_val, labels  = get_features(options, yamlConfig)

    #from models import model
    selmodel = getattr(models, yamlConfig['KerasModel'])
    model = selmodel(X_train_val.shape[1], Y_train_val.shape[1], l1Reg=yamlConfig['L1Reg'] )
    if not options.noCUDA and torch.cuda.is_available():
        model.cuda()

    ## Define loss and optimizer
    learning_rate=0.01
    # Nepoch = 6
    Nepoch = 600
    loss_fn = nn.BCELoss(size_average=False)
    Optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    RunWithDataLoader=False

    if RunWithDataLoader:
        dataset = TorchDataset(X_train_val, Y_train_val)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=1028,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=1)
        for epoch in range(Nepoch):
            for batch_number, (X_train, Y_train) in enumerate(data_loader):
                X_train = tensor_toVar(X_train)
                Y_train = tensor_toVar(Y_train, False)
                y_pred = model(X_train)
                loss = loss_fn(y_pred, Y_train)
                print(epoch, batch_number, loss.data[0])
                Optimizer.zero_grad()
                loss.backward()
                Optimizer.step()
    else:
        # Convert Numpy to tensor
        X_train = numpy_toVar(X_train_val)
        Y_train = numpy_toVar(Y_train_val, grad=False )
        for epoch in range(Nepoch):
            y_pred = model(X_train)
            loss = loss_fn(y_pred, Y_train)
            print(epoch, loss.data[0])
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

#============================================================================#
#--------------------------     Test The Model     --------------------------#
#============================================================================#
    X_test = numpy_toVar(X_test_val)
    Y_test = numpy_toVar(Y_test_val)
    output = model(X_test).data.cpu().numpy()
    Noutputs = Y_test.shape[1]
    output_train= model(X_train).data.cpu().numpy()

    for i in range(Noutputs):
        dfpr, dtpr, dthreshold = roc_curve(Y_test_val[:, i], output[:, i])
        test_auc = auc(dfpr, dtpr)
        tfpr, ttpr, tthreshold = roc_curve(Y_train_val[:,i], output_train[:,i])
        train_auc = auc(tfpr, ttpr)
        print("Tag %d, test AUC %f, train AUC %f " % (i, test_auc, train_auc))
