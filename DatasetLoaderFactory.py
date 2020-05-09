#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:01:17 2019

@author: daif
"""
import pandas as pd
from Utils import load_csv_dataset, rep_sample
from ConfigParser import ConfigsLoader


class ObjectFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)
    

class WikipediaTitleDatasetLoader:
    def __init__(self,configParser):
        self.configs = configParser
        self.read_config_variables()
        self.data = None
        return
        
    def read_config_variables(self):
        self.col_names = self.configs.get("DATA","col_names")
        self.limit_data = self.configs.getboolean("DEBUG","limit_data")
        self.data_limit = self.configs.getint("DEBUG","data_limit")
        self.sentence_column = self.configs.get("DATA","sentence_ column")
        self.target_column = self.configs.get("DATA","target_column")
        self.training_dataset_file_name = self.configs.get("DATA","training_csv_file_name")
        self.testing_dataset_file_name = self.configs.get("DATA","testing_csv_file_name")
        
    def read_dataset_from_file(self):
        print("Training dataset file: ")
        print(self.training_dataset_file_name)
        print("Testing dataset file: ")
        print(self.testing_dataset_file_name)
        self.training_data = pd.read_csv(self.training_dataset_file_name,names = self.col_names, skiprows = 1, encoding = "utf-8", sep = '\t')
        self.testing_data = pd.read_csv(self.testing_dataset_file_name,names = self.col_names, skiprows = 1, encoding = "utf-8", sep = '\t')
        #print("data columns: ", self.data.head())
        return
    def get_data(self):
        return self.training_data, self.testing_data
        
    def preprocess_dataset(self):
        self.training_data = self.preproces_target_column(self.training_data)
        self.testing_data = self.preproces_target_column(self.testing_data)

    def preproces_target_column(self, data):
        data[self.target_column] = data[self.target_column].astype(str)
        data[self.target_column] = pd.Categorical(data[self.target_column])
        data[self.target_column] = data[self.target_column].cat.codes
        return data
        
class PoetryDatasetLoader:
    def __init__(self,configParser):
        self.configs = configParser
        self.read_config_variables()
        self.data = None
    def read_config_variables(self):
        self.col_names = self.configs.get("DATA","col_names")
        self.limit_data = self.configs.getboolean("DEBUG","limit_data")
        self.data_limit = self.configs.getint("DEBUG","data_limit")
        self.sentence_column = self.configs.get("DATA","sentence_ column")
        self.target_column = self.configs.get("DATA","target_column")
        self.training_dataset_file_name = self.configs.get("DATA","training_csv_file_name")
        self.testing_dataset_file_name = self.configs.get("DATA","testing_csv_file_name")
            
    def read_dataset_from_file(self):
        print("Training dataset file: ")
        print(self.training_dataset_file_name)
        print("Testing dataset file: ")
        print(self.testing_dataset_file_name)
        self.training_data = pd.read_csv(self.training_dataset_file_name,names = self.col_names, skiprows = 1, encoding = "utf-8",lineterminator='\n')
        self.testing_data = pd.read_csv(self.testing_dataset_file_name,names = self.col_names, skiprows = 1, encoding = "utf-8",lineterminator='\n')
        return
    def get_data(self):
        return self.training_data, self.testing_data
    
    def preprocess_dataset(self):
        self.training_data = self.preproces_target_column(self.training_data)
        self.testing_data = self.preproces_target_column(self.testing_data)
        
    def preproces_target_column(self, data):
        data[self.target_column] = data[self.target_column].astype(str)
        data[self.target_column] = pd.Categorical(data[self.target_column])
        #print(self.training_data[self.target_column])
        data[self.target_column] = data[self.target_column].cat.codes
        return data

class DialectDatasetLoader:
    def __init__(self,configParser):
        self.configs = configParser
        self.read_config_variables()
        self.data = None
    def read_config_variables(self):
        self.col_names = self.configs.get("DATA","col_names")
        self.limit_data = self.configs.getboolean("DEBUG","limit_data")
        self.data_limit = self.configs.getint("DEBUG","data_limit")
        self.sentence_column = self.configs.get("DATA","sentence_ column")
        self.target_column = self.configs.get("DATA","target_column")
        self.training_dataset_file_name = self.configs.get("DATA","training_csv_file_name")
        self.testing_dataset_file_name = self.configs.get("DATA","testing_csv_file_name")
            
    def read_dataset_from_file(self):
        print("Training dataset file: ")
        print(self.training_dataset_file_name)
        print("Testing dataset file: ")
        print(self.testing_dataset_file_name)
        self.training_data = pd.read_csv(self.training_dataset_file_name,names = self.col_names, skiprows = 1, encoding = "utf-8",lineterminator='\n')
        self.testing_data = pd.read_csv(self.testing_dataset_file_name,names = self.col_names, skiprows = 1, encoding = "utf-8",lineterminator='\n')
        return
    def get_data(self):
        return self.training_data, self.testing_data
    
    def preprocess_dataset(self):
        self.training_data = self.preproces_target_column(self.training_data)
        self.testing_data = self.preproces_target_column(self.testing_data)
        
    def preproces_target_column(self, data):
        data[self.target_column] = data[self.target_column].astype(str)
        data[self.target_column] = pd.Categorical(data[self.target_column])
        #print(self.training_data[self.target_column])
        data[self.target_column] = data[self.target_column].cat.codes
        return data
    
def test_data_factory():
    configFileName = "Configs.ini"
    configLoader = ConfigsLoader(configFileName)
    configs = configLoader.get_configs()
    factory = ObjectFactory()
    factory.register_builder("Wikipedia_Title_Dataset",WikipediaTitleDatasetLoader)
    factory.register_builder("Poetry_Dataset",PoetryDatasetLoader)
    factory.register_builder("Dialect_Dataset",DialectDatasetLoader)
    poetry_loader = factory.create("Poetry_Dataset", **{'configParser':configs})
    poetry_loader.read_dataset_from_file()
    poetry_loader.preprocess_dataset()
    training_data, testing_data = poetry_loader.get_data()
    print(training_data.head())
    print(training_data.columns.values)
    
    print(testing_data.head())
    print(testing_data.columns.values)
    
    
#test_data_factory()
