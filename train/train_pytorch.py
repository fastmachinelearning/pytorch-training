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
from Data_loader import numpy_toVar, TorchDataset

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

    ## Convert Numpy to tensor
    X_train = numpy_toVar(X_train_val)
    Y_train = numpy_toVar(Y_train_val, grad=False )

    # dataset = TorchDataset(X_train_val, Y_train_val)
    # data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              # batch_size=-1,
                                              # shuffle=False,
                                              # num_workers=1)

    # print("not in GPU?")

    ## Define loss and optimizer
    learning_rate=0.01
    loss_fn = nn.BCELoss(size_average=False)
    Optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(600):
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
    output = model(X_test)
    dfpr, dtpr, dthreshold = roc_curve(Y_test.data.cpu().numpy(), output.data.cpu().numpy())
    dauc = auc(dfpr, dtpr)
    dfpr, dtpr, dthreshold = roc_curve(Y_train.data.cpu().numpy(), model(X_train).data.cpu().numpy())
    print (auc(dfpr, dtpr), dauc)
