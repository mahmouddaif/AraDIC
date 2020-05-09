#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:24:39 2018

@author: daif
"""
import numpy as np



from Datasets.TextDatasetCharToImage import TextDatasetCharToImage


class TextProcessor:
    def __init__(self,configParser):
        self.configParser = configParser
        self.initialize_variables()
        
    def initialize_variables(self):
        self.image_width = self.configParser.getint("TEXT_TRANSFORMATION","image_width")
        self.image_height = self.configParser.getint("TEXT_TRANSFORMATION","image_height")
        self.font_size = self.configParser.getint("TEXT_TRANSFORMATION","font_size")
        self.current_encoding = self.configParser.get("TEXT_TRANSFORMATION","current_encoding")
        self.fntFile = self.configParser.get("DATA","font_file")
        self.train_size = self.configParser.getfloat("TRAINING","train_percent")
        self.sentence_column = self.configParser.get("DATA","sentence_ column")
        self.target_column = self.configParser.get("DATA","target_column")
    

    def split_data_x_y(self, df):
        x = df[self.sentence_column]
        y = df[self.target_column]        
        x = x.reindex().values.astype(str)
        y = y.reindex().values.astype(np.int32)
        return x, y
    def numpy_pair_to_pytorch_dataset(self, x, y, chars_df, x_train):
        dataset =  TextDatasetCharToImage(TextDatasetCharToImage(self.configParser, x, y, chars_df))
        return dataset
    

