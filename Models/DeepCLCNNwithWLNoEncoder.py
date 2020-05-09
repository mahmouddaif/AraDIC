#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:20:21 2019

@author: daif
"""
import torch.nn.functional as F

from Models.DeepCLCNNNoEncoder import DeepCLCNNNoEncoder


class DeepCLCNNwithWLNoEncoder(DeepCLCNNNoEncoder):

    def __init__(self,
                 configParser,
                 out_dim,
                 **kwargs):
        self.configParser = configParser
        self.read_config_variables()
        super(DeepCLCNNwithWLNoEncoder, self).__init__(
            configParser, out_dim, **kwargs)

    def forward(self, x, **kwargs):
        h = F.dropout(x, p=self.wildcard_ratio, training=self.training)
        h = self.pool1(F.relu(self.conv1(x)))
        h = self.pool2(F.relu(self.conv2(h)))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        self.h_cam = h
        h = h.view(-1, self.flat_fts)
        h = self.fc1(h)
        h = self.fc2(h)
        return h

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
