#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:37:58 2019

@author: daif
"""

import torch.nn as nn
from Models.DeepNoPoolingCLCNN import DeepNoPoolingCLCNN
import torch.nn.functional as F


class DeepNoPoolingCLCNNwithWT(DeepNoPoolingCLCNN):

    def __init__(self,
                 configParser,
                 out_dim,
                 **kwargs):
        self.configParser = configParser
        self.read_config_variables()
        self.f = self.feat_maps
        self.ksize = self.ksize
        self.h_cam = None
        super(DeepNoPoolingCLCNNwithWT, self).__init__(
            configParser, out_dim, **kwargs)

        self.dropout = nn.Dropout2d(p=self.wildcard_ratio, inplace=True)

    def forward(self, x, **kwargs):

        x = self.encoder(x)

        x = F.dropout(x)  # apply wildcard training

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
