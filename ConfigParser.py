#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:02:55 2018

@author: daif
"""

import configparser
import datetime

class ConfigsLoader:
    def __init__(self, configFileName):
        self.config = configparser.RawConfigParser()
        self.config.read(configFileName)
        self.preprocess_column_names()
        self.preprocess_filter_sizes()
        #self.preprocess_kernel_size()
        
    def test(self):
        print(self.config)
        print(self.config.sections())
    
    def get_configs(self):
        return self.config        
    def preprocess_column_names(self):
        col_names = self.config.get("DATA","col_names")
        col_names =  col_names.split(",")
        col_names = [x.strip().lower() for x in col_names]
        self.config.set("DATA","col_names", col_names)
    def preprocess_logs_directory(self):
        currentDateTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trainer_logs_director = self.config.get("TRAINING","trainer_logs") + currentDateTime + "/"
        self.config.set("TRAINING", "trainer_logs", trainer_logs_director)
        return
    def preprocess_kernel_size(self):
        k_size = self.config.get("CECLCNN","ksize")
        k_size = k_size.split(",")
        k_size = [x.strip().lower() for x in k_size]
        self.config.set("CECLCNN", "ksize", (int(k_size[0]), int(k_size[1])))
        return
    def preprocess_filter_sizes(self):
        filter_sizes = self.config.get("CNN_Sentence","FILTER_SIZES")
        filter_sizes = filter_sizes.split(",")
        filter_sizes = [int(x.strip().lower()) for x in filter_sizes]
        self.config.set("CNN_Sentence", "FILTER_SIZES", filter_sizes)
        return
        
    
def test_configsLoader():
    configFileName = "Configs.ini"
    configLoader = ConfigsLoader(configFileName)
    
    configLoader.test()
    
    configs = configLoader.get_configs()
    print(configs.getint("TEXT_TRANSFORMATION","start"))
    
    
#test_configsLoader()