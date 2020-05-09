#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 17:10:19 2018

@author: daif
"""

from ConfigParser import ConfigsLoader
from Transformations import TextProcessor
import time
from Models.DeepCLCNN import DeepCLCNN
from Models.DeepCLCNNwithWL import  DeepCLCNNwithWL
from Models.DeepNoPoolingCLCNN import  DeepNoPoolingCLCNN
from Models.DeepNoPoolingCLCNNwithWT import DeepNoPoolingCLCNNwithWT
from Models.DeepCLCNNNoEncoder import DeepCLCNNNoEncoder
from Models.DeepCLCNNwithWLNoEncoder import DeepCLCNNwithWLNoEncoder
from Models.DeepNoPoolingCLCNNNoEncoder import DeepNoPoolingCLCNNNoEncoder
from Models.DeepNoPoolingCLCNNwithWTNoEncoder import DeepNoPoolingCLCNNwithWTNoEncoder
from Models.RNNCharacter import RNNCharacter

from TrainerWrapper import TrainerWrapper
from Reporter import Reporter
import numpy as np
from DatasetLoaderFactory import ObjectFactory, WikipediaTitleDatasetLoader, PoetryDatasetLoader, DialectDatasetLoader
from Utils import load_csv_dataset,construct_count_array, load_model
from TextCleaner import TextCleaner

class MainProgram:
    def __init__(self,configFileName):
        self.start = time.time()
        configLoader = ConfigsLoader(configFileName)
        self.configs = configLoader.get_configs()
        self.textProcessor = TextProcessor(self.configs)
        self.trainerWrapper = TrainerWrapper(self.configs)
        self.reporter = Reporter(self.configs)
        self.text_cleaner = TextCleaner(self.configs)
        self.read_config_variables()
    def read_config_variables(self):
        self.col_names = self.configs.get("DATA","col_names")
        self.limit_data = self.configs.getboolean("DEBUG","limit_data")
        self.data_limit = self.configs.getint("DEBUG","data_limit")
        self.current_dataset = self.configs.get("DATA","current_dataset")
        self.target_column = self.configs.get("DATA","target_column")
        self.do_preprocessing = self.configs.getboolean("PREPROCESSING","do_preprocessing")
        self.loss_function_name = self.configs.get("TRAINING","loss_function")
        self.balanced_loss = self.configs.getboolean("TRAINING","balanced_loss")
        self.chars_file_name = self.configs.get("DATA","chars_csv_file_name")
        self.char_df_colNames = [self.configs.get("DATA","char_df_column")]
        self.device_id = self.configs.get("TRAINING","device_id")
        self.current_network  = self.configs.get("ARCHITECTURE","current_network")
        self.trainer_logs_out = self.configs.get("TRAINING","trainer_logs")
        self.checkpoint_file = self.configs.get("DATA","checkpoint_file")
        
        self.model_file = self.trainer_logs_out + self.checkpoint_file + 'ckpt.t7'
        
    def load_image_dataset(self):
        self.chars_df = load_csv_dataset(self.chars_file_name, ",", self.char_df_colNames)
        factory = ObjectFactory()
        factory.register_builder("Wikipedia_Title_Dataset",WikipediaTitleDatasetLoader)
        factory.register_builder("Poetry_Dataset",PoetryDatasetLoader)
        factory.register_builder("Dialect_Dataset",DialectDatasetLoader)
        data_loader = factory.create(self.current_dataset, **{'configParser':self.configs})
        data_loader.read_dataset_from_file()
        data_loader.preprocess_dataset()
        self.training_data_raw, self.testing_data_raw = data_loader.get_data()
        if self.limit_data:
            self.training_data_raw = self.training_data_raw.groupby(self.target_column, group_keys=False).apply(lambda x: x.sample(int(np.rint(self.data_limit*len(x)/len(self.training_data_raw))))).sample(frac=1).reset_index(drop=True)

        if self.do_preprocessing:
            self.training_data_raw = self.text_cleaner.clean_text(self.training_data_raw)
            self.testing_data_raw = self.text_cleaner.clean_text(self.testing_data_raw)
        self.label_list = sorted(list(set(self.training_data_raw[self.target_column])))
        self.counts = construct_count_array(self.training_data_raw[self.target_column])
        self.labels = self.training_data_raw[self.target_column]
        self.number_of_classes = np.size(np.unique(self.labels))
        self.X_train, self.y_train = self.textProcessor.split_data_x_y(self.training_data_raw)
        self.X_test, self.y_test = self.textProcessor.split_data_x_y(self.testing_data_raw)
        
        self.training_data = self.textProcessor.numpy_pair_to_pytorch_dataset(self.X_train, self.y_train,self.chars_df, self.X_train)
        self.testing_data = self.textProcessor.numpy_pair_to_pytorch_dataset(self.X_test, self.y_test,self.chars_df, self.X_train)
        print(self.label_list)
        return self.training_data, self.testing_data

            
    
    def build_model(self, out_dim):
        factory = ObjectFactory()
        factory.register_builder("DeepCLCNN",DeepCLCNN)
        factory.register_builder("DeepCLCNNwithWL",DeepCLCNNwithWL)
        factory.register_builder("DeepNoPoolingCLCNN",DeepNoPoolingCLCNN)
        factory.register_builder("DeepNoPoolingCLCNNwithWT",DeepNoPoolingCLCNNwithWT)
        
        factory.register_builder("DeepCLCNNNoEncoder",DeepCLCNNNoEncoder)
        factory.register_builder("DeepCLCNNwithWLNoEncoder",DeepCLCNNwithWLNoEncoder)
        factory.register_builder("DeepNoPoolingCLCNNNoEncoder",DeepNoPoolingCLCNNNoEncoder)
        factory.register_builder("DeepNoPoolingCLCNNwithWTNoEncoder",DeepNoPoolingCLCNNwithWTNoEncoder)
        factory.register_builder("RNNCharacter",RNNCharacter)

        model = factory.create(self.current_network, **{'configParser':self.configs, 'out_dim':out_dim})
        return model
    def run(self):
        training_data, testing_data = self.load_image_dataset()
        self.model = self.build_model(len(self.label_list))
        self.model.cuda()
    
        self.trainerWrapper.train_model(self.model,training_data, testing_data, self.number_of_classes, self.y_train, self.counts)
        self.end = time.time()
        print("Training took %s seconds!",str(self.end - self.start))
        print("Now loading the model with the best performance")
        self.model = self.build_model(len(self.label_list)).cuda()
        #self.model = load_model(self.model_file, DistributedDataParallel(self.model))
        self.model = load_model(self.model_file,self.model)
        print("Now testing no sliding window ....")
        self.start = time.time()
        self.reporter.test_model(self.model, testing_data, self.number_of_classes)
        self.end = time.time()
        print("Testing no sliding window took %s seconds!",str(self.end - self.start))
        print("Now testing Sliding Window...")
        print("Printing the class distripution in the whole data")
        self.reporter.print_category_distribution(self.training_data_raw[self.target_column])
        self.reporter.write_results()

        
        
        
mainProgram = MainProgram('Configs.ini')
mainProgram.run()
