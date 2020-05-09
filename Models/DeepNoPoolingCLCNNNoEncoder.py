#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Nov 21 16:31:29 2019

@author: daif
"""

import torch
import torch.nn as nn

from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class DeepNoPoolingCLCNNNoEncoder(nn.Module):

    def __init__(self,
                 configParser,
                 out_dim,
                 **kwargs):
        self.configParser = configParser
        self.read_config_variables()
        self.f = self.feat_maps
        self.ksize = self.ksize
        self.h_cam = None
        super(DeepNoPoolingCLCNNNoEncoder, self).__init__()
        self.conv1 = nn.Conv1d(self.encode_dim, self.f, self.ksize, stride=3)
        self.conv2 = nn.Conv1d(self.f, self.f, self.ksize, stride=3)
        self.conv3 = nn.Conv1d(self.f, self.f, self.ksize)
        self.conv4 = nn.Conv1d(self.f, self.f, self.ksize)
        self.conv5 = nn.Conv1d(self.f, self.f, self.ksize)
        self.flat_fts = self.get_flat_fts()
        self.fc1 = nn.Linear(self.flat_fts, self.fcl_size)
        self.fc2 = nn.Linear(self.fcl_size, self.fcl_size)
        self.fc3 = nn.Linear(self.fcl_size, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def get_flat_fts(self):
        in_size = (self.encode_dim, self.c)
        x = Variable(torch.ones(1, *in_size))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        flat_fts = int(np.prod(x.size()[1:]))
        return flat_fts

    def forward(self, x, **kwargs):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        self.h_cam = x

        x = x.view(-1, self.flat_fts)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def read_config_variables(self):
        self.wildcard_ratio = self.configParser.getfloat(
            "CECLCNN", "wildcard_ratio")
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
