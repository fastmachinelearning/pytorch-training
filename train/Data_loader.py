#!/usr/bin/env python3
# encoding: utf-8

# File        : Data_loader.py
# Author      : Ben Wu
# Contact     : benwu@fnal.gov
# Date        : 2018 Apr 05
#
# Description : Function for handling data loading and preprocessing

import h5py
import yaml
import torch
import pandas as pd
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def get_features(options, yamlConfig):
    # To use one data file:
    h5File = h5py.File(options.inputFile)
    treeArray = h5File[options.tree][()]

    # List of features to use
    features = yamlConfig['Inputs']

    # List of labels to use
    labels = yamlConfig['Labels']

    # Convert to dataframe
    features_df = pd.DataFrame(treeArray,columns=features)
    labels_df = pd.DataFrame(treeArray,columns=labels)
    if 'Conv' in yamlConfig['InputType']:
        labels_df = labels_df.drop_duplicates()

    # Convert to numpy array
    features_val = features_df.values
    labels_val = labels_df.values
    if 'Conv' in yamlConfig['InputType']:
        labels_val = labels_val[:,:-1] # drop the last label j_pt
        # print labels_val.shape

    if yamlConfig['InputType']=='Conv1D':
        features_2dval = np.zeros((len(labels_df), yamlConfig['MaxParticles'], len(features)-1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[features_df['j_pt']==labels_df['j_pt'].iloc[i]]
            index_values = features_df_i.index.values
            #features_val_i = features_val[index_values[0]:index_values[-1]+1,:-1] # drop the last feature j_pt
            features_val_i = features_val[np.array(index_values),:-1] # drop the last feature j_pt
            nParticles = len(features_val_i)
            if nParticles>yamlConfig['MaxParticles']:
                features_val_i =  features_val_i[0:yamlConfig['MaxParticles'],:]
            else:
                features_val_i = np.concatenate([features_val_i, np.zeros((yamlConfig['MaxParticles']-nParticles, len(features)-1))])

            features_2dval[i, :, :] = features_val_i

        features_val = features_2dval

    elif yamlConfig['InputType']=='Conv2D':
        features_2dval = np.zeros((len(labels_df), yamlConfig['BinsX'], yamlConfig['BinsY'], 1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[features_df['j_pt']==labels_df['j_pt'].iloc[i]]
            index_values = features_df_i.index.values

            xbins = np.linspace(yamlConfig['MinX'],yamlConfig['MaxX'],yamlConfig['BinsX']+1)
            ybins = np.linspace(yamlConfig['MinY'],yamlConfig['MaxY'],yamlConfig['BinsY']+1)

            x = features_df_i[features[0]]
            y = features_df_i[features[1]]
            w = features_df_i[features[2]]

            hist, xedges, yedges = np.histogram2d(x, y, weights=w, bins=(xbins,ybins))

            for ix in range(0,yamlConfig['BinsX']):
                for iy in range(0,yamlConfig['BinsY']):
                    features_2dval[i,ix,iy,0] = hist[ix,iy]
        features_val = features_2dval

    X_train_val, X_test, y_train_val, y_test = train_test_split(features_val, labels_val, test_size=0.2, random_state=42)

    #Normalize inputs
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType']!='Conv1D' and yamlConfig['InputType']!='Conv2D':
        scaler = preprocessing.StandardScaler().fit(X_train_val)
        X_train_val = scaler.transform(X_train_val)
        X_test = scaler.transform(X_test)

    #Normalize conv inputs
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType']=='Conv1D':
        reshape_X_train_val = X_train_val.reshape(X_train_val.shape[0]*X_train_val.shape[1],X_train_val.shape[2])
        scaler = preprocessing.StandardScaler().fit(reshape_X_train_val)
        for p in range(X_train_val.shape[1]):
            X_train_val[:,p,:] = scaler.transform(X_train_val[:,p,:])
            X_test[:,p,:] = scaler.transform(X_test[:,p,:])

    if 'Conv' in yamlConfig['InputType']:
        labels = labels[:-1]

    return X_train_val, X_test, y_train_val, y_test, labels

## Config module
def parse_config(config_file) :
    print ("Loading configuration from " + str(config_file))
    config = open(config_file, 'r')
    return yaml.load(config)

def tensor_toVar(npval, grad=True):
    val_tensor = npval.float()
    if torch.cuda.is_available():
        val_tensor = val_tensor.cuda()
    return Variable(val_tensor, requires_grad=grad)

def numpy_toVar(npval, grad=True):
    val_tensor = torch.from_numpy(npval)
    return tensor_toVar(val_tensor, grad)

class TorchDataset(Dataset):
    def __init__(self, X_train, Y_train, transform=None ):
        super().__init__()
        self.xtrain = X_train
        self.ytrain = Y_train
        self.transform = transform

    def __len__(self):
        return len(self.xtrain)

    def __getitem__(self, idx):
        xtrain = self.xtrain[idx]
        ytrain = self.ytrain[idx]

        if self.transform:
            xtrain = self.transform(xtrain)
            ytrain = self.transform(ytrain)

        return xtrain, ytrain

