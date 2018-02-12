import sys
import os
import yaml
import h5py
import models
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
from optparse import OptionParser
from torch.autograd import Variable
from torch import nn

sys.path.insert(0, "../../keras-training/train/")
from train import get_features

## Config module
def parse_config(config_file) :
    print "Loading configuration from " + str(config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-i','--input'   ,action='store',type='string',dest='inputFile'   ,default='../data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z', help='input file')
    parser.add_option('-t','--tree'   ,action='store',type='string',dest='tree'   ,default='t_allpar_new', help='tree name')
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
    parser.add_option('-c','--config'   ,action='store',type='string', dest='config', default='train_config_twolayer.yml', help='configuration file')
    (options,args) = parser.parse_args()

    yamlConfig = parse_config(options.config)

    if os.path.isdir(options.outputDir):
        os.removedirs(options.outputDir)
    else:
        os.mkdir(options.outputDir)

    ## Getting the train and test sample
    X_train_val, X_test_val, Y_train_val, Y_test_val, labels  = get_features(options, yamlConfig)
    ## Convert Numpy to tensor
    X_train_tensor = torch.from_numpy(X_train_val)
    X_test_tensor  = torch.from_numpy(X_test_val)
    Y_train_tensor = torch.from_numpy(Y_train_val)
    Y_test_tensor  = torch.from_numpy(Y_test_val)

    ## Prepare variables for training
    X_train = Variable(X_train_tensor)
    Y_train = Variable(Y_train_tensor.float(), requires_grad=False)

    #from models import model
    selmodel = getattr(models, yamlConfig['KerasModel'])
    model = selmodel(X_train_tensor.shape[1], Y_train_tensor.shape[1], l1Reg=yamlConfig['L1Reg'] )

    ## Define loss and optimizer
    learning_rate=0.01
    loss_fn = nn.BCELoss(size_average=False)
    Optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for i in range(600):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, Y_train)
        print(i, loss.data[0])
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()


    # Test the Model
    correct = 0
    total = 0
    X_test = Variable(X_test_tensor)
    output = model(X_test)
    dfpr, dtpr, dthreshold = roc_curve(Y_test_tensor.numpy(), output.data.numpy())
    dauc = auc(dfpr, dtpr)
    dfpr, dtpr, dthreshold = roc_curve(Y_train_tensor.numpy(), model(X_train).data.numpy())
    print auc(dfpr, dtpr), dauc
