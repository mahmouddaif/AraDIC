#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 02:20:47 2019

@author: daif
"""

#from Main import MainProgram
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import torch 
import collections
from ConfigParser import ConfigsLoader
import time
from DatasetLoaderFactory import ObjectFactory, WikipediaTitleDatasetLoader, PoetryDatasetLoader
from Transformations import TextProcessor
from Utils import construct_df_from_dict, append_two_dataframes
import os.path



class Reporter:
    def __init__(self,configParser):
        self.configParser = configParser
        self.read_config_variables()
        self.textProcessor = TextProcessor(self.configParser)
        self.results_dict = self.build_initial_results_dictionary()
    def read_config_variables(self):
        self.device_id = self.configParser.get("TRAINING","device_id")
        self.n_processes = self.configParser.getint("TESTING","n_processes")
        self.batch_size = self.configParser.getint("TRAINING","batch_size")
        self.sentence_start = self.configParser.getint("TEXT_TRANSFORMATION","start")
        self.sentence_end = self.configParser.getint("TEXT_TRANSFORMATION","end")
        self.sliding_window_stride = self.configParser.getint("TEXT_TRANSFORMATION","sliding_window_stride")
        self.sentence_column = self.configParser.get("DATA","sentence_ column")
        self.target_column = self.configParser.get("DATA","target_column")
        self.train_percent = self.configParser.getfloat("TRAINING","train_percent")
        self.epoch = self.configParser.getint("TRAINING","epoch")
        self.current_dataset = self.configParser.get("DATA","current_dataset")
        self.weight_decay = self.configParser.getfloat("TRAINING","weight_decay")
        self.beta = self.configParser.getfloat("TRAINING","beta")
        self.dropout_CE = self.configParser.getfloat("CE","dropout")
        self.dropout_CLNN = self.configParser.getfloat("CECLCNN","dropout")
        self.results_file = self.configParser.get("DATA","results_file")
        self.char_len = self.sentence_end - self.sentence_start
    def print_category_distribution(self, labels_array):
        multiplyer = 100.0/ len(labels_array)
        print("Printing the article distribution: ")
        counts = collections.Counter(labels_array)
        print(counts)
        #print("The type of the county variable! ")
        #print(type(counts))
        print("Now normalizing the counts: ")
        for key in counts:
            counts[key] = round(counts[key]*multiplyer,2)
        print(counts)
    def test_print_category_distribution(self):
        arr = [1,1,1,1,1,2,2,3,4,4,5,3,2,7,8,9,10,10,10,8, 3]
        arr = np.asarray(arr, dtype=np.float32)
        self.print_category_distribution(arr)
        
    def generate_subseq(self, body):
        subsequences = []
        for k in range(0, len(body) - self.char_len + 1, self.sliding_window_stride):
            subsequences.append(body[k:k + self.char_len])
        return subsequences

    
    def test_apply_sliding_window(self, x,y, model, number_of_classes,chars_df, x_train):
        model.eval()
        x = x.tolist()
        y = y.tolist()
        sentences = []
        labels = []
        test_results = {
                "y_true": [],
                "y_pred": [],
        }
        model.cuda()
        for i in range(len(x)):
            sentence = x[i]
            label = y[i]
            while len(sentence) < self.char_len:
                sentence += ' '
                
            subsequences = self.generate_subseq(sentence)
            sentences.extend(subsequences)
            if len(sentences) == 0:
                continue
            tmp_labels = [label for _ in range(0, len(sentence) -self.char_len + 1, self.sliding_window_stride)]
            labels.extend(tmp_labels)
            tmp_labels = np.asarray(tmp_labels)
            subsequences = np.asarray(subsequences)
            testing_data = self.textProcessor.numpy_pair_to_pytorch_dataset(subsequences, tmp_labels,chars_df,x_train)
            #print("Number of training data is: ", len(testing_data))
            testloader = torch.utils.data.DataLoader(
                    testing_data, 
                    batch_size = self.batch_size, 
                    shuffle = False, 
                    num_workers = self.n_processes,
                    collate_fn=testing_data.collate_fn)
            batch_outputs = np.zeros(number_of_classes, dtype=np.float32)
            
            with torch.no_grad():
                for i, batch in enumerate(testloader):
                    X_test, y_test,lengths = batch['sentences'].cuda(), batch['labels'].cuda(),batch['lengths'].cuda()
                    y_pred_batch = model(X_test,lengths)
                    # mean predictions
                    y_pred_batch = y_pred_batch.cpu().detach().numpy()
                    #print(y_pred_batch.shape)
                    batch_outputs += np.mean(y_pred_batch, axis=0)
            test_results["y_pred"].append(int(np.argmax(batch_outputs)))
            test_results["y_true"].append(label)
            #print("The sentence is: ", sentence)
            #print("The label is: ", label)
        print("Now printing the confusion matrix!")        
        confusion_matrix_test = confusion_matrix(test_results["y_true"], test_results["y_pred"])
        print(confusion_matrix_test)
        self.results_dict["validation_confusion_matrix_sliding_window"] = confusion_matrix_test
        print("Now printing the classification report")
        classification_report_test = classification_report(test_results["y_true"], test_results["y_pred"],digits=4)
        self.results_dict["validation_classification_report_sliding_window"] = classification_report_test
        print(classification_report_test)
        print("Now printing the distripution of each class in the test data")
        self.print_category_distribution(test_results["y_true"])
    
    def test_model(self, model, dataset, number_of_classes):
        model.eval()
        model.cuda()
            
        dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size = self.batch_size, 
                shuffle = False, 
                num_workers = self.n_processes,
                collate_fn=dataset.collate_fn)
        
        test_results = {
                "y_true": [],
                "y_pred": [],
        }
        

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                X_test, y_test,lengths = batch['sentences'].cuda(), batch['labels'].cuda(),batch['lengths'].cuda()
                y_pred_batch = model(X_test,lengths)
                
                #print("The y_pred_batch is: ")
                #print(y_pred_batch)
                y_pred_batch = np.argmax(y_pred_batch.cpu().detach().numpy(), axis = 1)
                #print("The shape of the y_pred_batch: ", y_pred_batch.shape)
                test_results["y_pred"].extend(y_pred_batch)
                test_results["y_true"].extend(y_test)
        test_results["y_pred"] = np.asarray(test_results["y_pred"], dtype=np.int32)
        test_results["y_true"] = np.asarray(test_results["y_true"], dtype=np.int32)
        print("The type of the predicted variable: ", type(test_results["y_pred"]))
        print("The type of the original variable: ",type(test_results["y_true"]))
        print("Now printing the size of the predicted variable: ", len(test_results["y_pred"]))
        print("Now printing the size of the original variable: ", len(test_results["y_true"]))
        print("Now printing the confusion matrix!") 
        confusion_matrix_test = confusion_matrix(test_results["y_true"], test_results["y_pred"])
        self.results_dict["validation_confusion_matrix_single_input"] = confusion_matrix_test
        print(confusion_matrix_test)
        print("Now printing the classification report")
        classification_report_test = classification_report(test_results["y_true"], test_results["y_pred"],digits=4)
        self.results_dict["validation_classification_report_single_input"] = classification_report_test
        print(classification_report_test)
        print("Now printing the distripution of each class in the test data")
        self.print_category_distribution(test_results["y_true"])
    def build_initial_results_dictionary(self):
        initial_results_dict = {}
        initial_results_dict["start"] = self.sentence_start
        initial_results_dict["end"] = self.sentence_end
        initial_results_dict["sliding_window_stride"] = self.sliding_window_stride
        initial_results_dict["current_dataset"] = self.current_dataset
        initial_results_dict["train_percent"] = self.train_percent
        initial_results_dict["batch_size"] = self.batch_size
        initial_results_dict["epoch"] = self.epoch
        initial_results_dict["weight_decay"] = self.weight_decay
        initial_results_dict["beta"] = self.beta
        initial_results_dict["balanced_iterator"] = False
        initial_results_dict["dropout_CE"] = self.dropout_CE
        initial_results_dict["dropout_CLNN"] = self.dropout_CLNN
        initial_results_dict["validation_classification_report_sliding_window"] = ""
        initial_results_dict["validation_confusion_matrix_sliding_window"] = ""
        initial_results_dict["validation_classification_report_single_input"] = ""
        initial_results_dict["validation_confusion_matrix_single_input"] = ""
        return initial_results_dict
    
    def write_results(self):
        print("The results file location, " ,self.results_file)
        current_results_df = construct_df_from_dict(self.results_dict)
        if os.path.isfile(self.results_file):
            print("Not first time to write to file!")
            results_df = pd.read_csv(self.results_file)
            results_df = append_two_dataframes(results_df, current_results_df)
            results_df.to_csv(self.results_file, index = False)
        else:
            print("First time to write")
            current_results_df.to_csv(self.results_file, index = False)
        
def test_reporter():
    configFileName = "Configs.ini"
    configLoader = ConfigsLoader(configFileName)
    configs = configLoader.get_configs()
    reporter = Reporter(configs)
    start = time.time()
    reporter.test_print_category_distribution()
    end = time.time()
    print("Getting distribution took %s seconds!",str(end - start))
#test_reporter()
def test_sliding_window():
    main = MainProgram('Configs.ini')
    textProcessor = TextProcessor(main.configs)
    factory = ObjectFactory()
    factory.register_builder("Wikipedia_Title_Dataset",WikipediaTitleDatasetLoader)
    factory.register_builder("Poetry_Dataset",PoetryDatasetLoader)
    data_loader = factory.create(main.current_dataset, **{'configParser':main.configs})
    data_loader.read_dataset_from_file()
    data_loader.preprocess_dataset()
    data = data_loader.get_data()
    X_train, X_test, y_train, y_test = textProcessor.split_training_testing(data)
    reporter = Reporter(main.configs)
    print("Type of testing data, ", type(X_test))
    new_testing_data = reporter.apply_sliding_window(X_test,y_test)
    print(new_testing_data)
    
#test_sliding_window()
