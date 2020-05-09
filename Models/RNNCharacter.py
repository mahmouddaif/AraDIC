#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:43:32 2020

@author: daif
"""

import torch
import torch.nn as nn
from Models.CharacterEncoder import CharacterEncoder



class RNNCharacter(nn.Module):
    def __init__(self,
                 configParser,
                 out_dim,
                 **kwargs):
        super(RNNCharacter, self).__init__()
        # embedding
        self.configParser = configParser
        self.read_config_variables()
        self.encoder = CharacterEncoder(self.configParser, self.embed_size)

        self.drop_en = nn.Dropout(p=self.wildcard_ratio)

        # rnn module
        if self.rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.1,
                                batch_first=True, bidirectional=True)
        elif self.rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.1,
                                batch_first=True, bidirectional=True)
        else:
            raise LookupError(' only support LSTM and GRU')


        self.bn2 = nn.BatchNorm1d(self.hidden_size*2)
        self.fc = nn.Linear(self.hidden_size*2, out_dim)
        
    def read_config_variables(self):
        self.rnn_model = self.configParser.get("GRU", "rnn_model")
        self.hidden_size = self.configParser.getint("GRU", "hidden_size")
        self.num_layers = self.configParser.getint("GRU", "num_layers")
        self.wildcard_ratio = self.configParser.getfloat("GRU", "wildcard_ratio")
        self.embed_size = self.configParser.getint("GRU", "embed_size")
        self.use_last = self.configParser.getboolean("GRU", "use_last")

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]

        x_embed = self.encoder.forward(x)

        x_embed = self.drop_en(x_embed)
        x_embed = x_embed.view(x_embed.shape[0], x_embed.shape[2], x_embed.shape[1])

        # None is for initial hidden state
        self.rnn.flatten_parameters()
        out_rnn, ht = self.rnn(x_embed, None)

        row_indices = torch.arange(0, batch_size).long()
        col_indices = torch.ones(x.shape[0])* (self.embed_size-1)
        if next(self.parameters()).is_cuda:
            row_indices = row_indices.cuda()
            col_indices = col_indices.cuda()

        if self.use_last:
            last_tensor=out_rnn[row_indices, col_indices, :]
        else:
            # use mean
            last_tensor = out_rnn[row_indices, :, :]
            last_tensor = torch.mean(last_tensor, dim=1)

        fc_input = self.bn2(last_tensor)
        out = self.fc(fc_input)
        return out
