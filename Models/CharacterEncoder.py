#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 23:11:25 2018

@author: daif
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CharacterEncoder(nn.Module):

    def __init__(self, configParser, encode_dim=128):
        super(CharacterEncoder, self).__init__()
        self.encode_dim = encode_dim
        self.configParser = configParser
        self.read_config_variables()

        self.conv1 = nn.Conv2d(self.chars, self.feature_maps, self.k_size)
        self.conv2 = nn.Conv2d(
            self.feature_maps, self.feature_maps, self.k_size)
        self.conv3 = nn.Conv2d(
            self.feature_maps, self.feature_maps, self.k_size)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)

        self.flat_fts = self.get_flat_fts()
        self.fc1 = nn.Linear(self.flat_fts, self.fc1_out)
        self.fc2 = nn.Linear(self.fc2_in, encode_dim)
        
    def get_flat_fts(self):
        in_size = (self.c, 1, self.img_h, self.img_w)
        h = Variable(torch.ones( *in_size))
        h = self.pool1(F.relu(self.conv1(h)))
        h = self.pool2(F.relu(self.conv2(h)))
        h = F.relu(self.conv3(h))

        flat_fts = int(np.prod(h.size()[1:]))
        return flat_fts


    def encode(self, x):

        h = self.pool1(F.relu(self.conv1(x)))
        h = self.pool2(F.relu(self.conv2(h)))
        h = F.relu(self.conv3(h))
        h = h.view(-1, self.flat_fts)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        return h

    def read_config_variables(self):
        self.chars = self.configParser.getint("CE", "chars")
        self.feature_maps = self.configParser.getint("CE", "feature_maps")
        self.fc1_in = self.configParser.getint("CE", "fc1_in")
        self.fc1_out = self.configParser.getint("CE", "fc1_out")
        self.fc2_in = self.configParser.getint("CE", "fc2_in")
        self.k_size = self.configParser.getint("CE", "k_size")
        self.dropout_ratio = self.configParser.getfloat("CE", "dropout")
        self.img_h = self.configParser.getint(
            "TEXT_TRANSFORMATION", "image_height")
        self.img_w = self.configParser.getint(
            "TEXT_TRANSFORMATION", "image_width")
        self.c = self.configParser.getint("CECLCNN", "num_chars")

    def forward(self, x,**kwargs):
        #print("X shape: ", x.shape)
        h = x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])
        h = self.encode(h)
        #print("Encoded Shape: ", h.shape)
        h = h.view(x.shape[0], self.encode_dim, x.shape[1])

        return h
