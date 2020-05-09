#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 18:42:00 2019

@author: daif
"""
from Clean_Arabic_Text import my_preprocessor

class TextCleaner:
    def __init__(self,configParser):
        self.configParser = configParser
        self.initialize_variables()
    
    def initialize_variables(self):
        self.sentence_column = self.configParser.get("DATA","sentence_ column")
        self.target_column = self.configParser.get("DATA","target_column")
        
    def clean_text(self, df):
        df[self.sentence_column] = df.apply(my_preprocessor,  sentence_column = self.sentence_column, axis = 1)
        #print("Modified Dataframe by applying lambda function on each row:")
        #print(df[self.sentence_column])
        return df

def test_text_cleaner():
    from ConfigParser import ConfigsLoader
    from DatasetLoaderFactory import ObjectFactory, WikipediaTitleDatasetLoader, PoetryDatasetLoader
    configFileName = "Configs.ini"
    configLoader = ConfigsLoader(configFileName)
    configs = configLoader.get_configs()
    text_cleaner = TextCleaner(configs)
    factory = ObjectFactory()
    factory.register_builder("Wikipedia_Title_Dataset",WikipediaTitleDatasetLoader)
    factory.register_builder("Poetry_Dataset",PoetryDatasetLoader)
    poetry_loader = factory.create("Poetry_Dataset", **{'configParser':configs})
    poetry_loader.read_dataset_from_file()
    poetry_loader.preprocess_dataset()
    df = poetry_loader.get_data()
    df = df[1:10]
    print("Number of columns before clean is: ", len(df.columns.values))
    print("Columns before clean are: ", df.columns.values)
    df.to_csv("data/test_clean_dat_before.csv", encoding = "utf-8", index = False)
    print("Datafram before cleaning: ",df["title"])
    df = text_cleaner.clean_text(df)
    print("Number of columns after clean is: ", len(df.columns.values))
    print("Columns after clean: ", df.columns.values)
    print("Dataframe after cleaning: ", df["title"])
    df.to_csv("data/test_clean_dat_after.csv", encoding = "utf-8", index = False)
    
#test_text_cleaner()