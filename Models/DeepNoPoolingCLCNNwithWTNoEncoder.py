#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:40:23 2019

@author: daif
"""

import torch.nn.functional as F


from Models.DeepNoPoolingCLCNNNoEncoder import DeepNoPoolingCLCNNNoEncoder


class DeepNoPoolingCLCNNwithWTNoEncoder(DeepNoPoolingCLCNNNoEncoder):

    def __init__(self,
                 configParser,
                 out_dim,
                 **kwargs):
        self.configParser = configParser
        self.read_config_variables()
        self.f = self.feat_maps
        self.ksize = self.ksize
        self.h_cam = None
        super(DeepNoPoolingCLCNNNoEncoder, self).__init__(
            configParser, out_dim, **kwargs)

    def forward(self, x, **kwargs):

        x = F.dropout(x, p=self.wildcard_ratio, training=self.training)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        self.h_cam = x

        x = x.view(-1, self.flat_fts)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
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
