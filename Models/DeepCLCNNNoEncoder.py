#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:08:22 2019

@author: daif
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np


class DeepCLCNNNoEncoder(nn.Module):

    def __init__(self,
                 configParser,
                 out_dim,
                 **kwargs):
        self.configParser = configParser
        self.read_config_variables()
        self.f = self.feat_maps
        self.ksize = self.ksize
        self.h_cam = None

        super(DeepCLCNNNoEncoder, self).__init__()

        self.conv1 = nn.Conv1d(self.encode_dim, self.f, self.ksize)
        self.conv2 = nn.Conv1d(self.f, self.f, self.ksize)
        self.conv3 = nn.Conv1d(self.f, self.f, self.ksize)
        self.conv4 = nn.Conv1d(self.f, self.f, self.ksize)
        #self.conv5 = nn.Conv2d(self.f, self.f, self.ksize)
        self.pool1 = nn.MaxPool1d(3)
        self.pool2 = nn.MaxPool1d(3)

        self.flat_fts = self.get_flat_fts()
        self.fc1 = nn.Linear(self.flat_fts, self.fcl_size)
        self.fc2 = nn.Linear(self.fcl_size, out_dim)

    def get_flat_fts(self):
        in_size = (self.encode_dim, self.c)
        h = Variable(torch.ones(1, *in_size))
        # print(h.shape)
        #import pdb;pdb.set_race()
        h = self.pool1(F.relu(self.conv1(h)))
        # print(h.shape)
        h = self.pool2(F.relu(self.conv2(h)))
        # print(h.shape)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        flat_fts = int(np.prod(h.size()[1:]))
        return flat_fts

    def forward(self, x, **kwargs):
        #print("Original input size: ")
        # print(x.size())
        #h = self.encoder.forward(x)
        #import pdb;pdb.set_race()
        #print("Original input size: ")
        h = self.pool1(F.relu(self.conv1(x)))
        #print("Shape after first convolution: ")
        # print(h.size())
        h = self.pool2(F.relu(self.conv2(h)))
        #print("Shape after second convolution: ")
        # print(h.size())
        h = F.relu(self.conv3(h))
        #print("Shape after third convolution: ")
        # print(h.size())

        h = F.relu(self.conv4(h))
        #print("Shape after fourth convolution: ")
        # print(h.size())
        # pdb.set_trace()
        # print(h.size())
        #h = F.relu(self.conv5(h))
        self.h_cam = h
        h = h.view(-1, self.flat_fts)

        h = self.fc1(h)
        h = self.fc2(h)

        return h

    def read_config_variables(self):
        self.encode_dim = self.configParser.getint("CECLCNN", "char_enc_dim")
        self.feat_maps = self.configParser.getint("CECLCNN", "feature_maps")
        self.ksize = self.configParser.getint("CECLCNN", "ksize")
        self.fcl_size = self.configParser.getint("CECLCNN", "fc_layer_size")
        self.dropout_ratio = self.configParser.getfloat("CECLCNN", "dropout")
        self.c = self.configParser.getint("CECLCNN", "num_chars")
        self.current_encoder = self.configParser.get("CE", "current_encoder")
        self.img_h = self.configParser.getint(
            "TEXT_TRANSFORMATION", "image_height")
        self.img_w = self.configParser.getint(
            "TEXT_TRANSFORMATION", "image_width")
