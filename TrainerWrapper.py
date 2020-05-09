#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 19:44:12 2019

@author: daif
"""
from Utils import create_directory_if_not_exists
import torch
import torch.optim as optim
import torch.nn as nn
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm.autonotebook import tqdm
import inspect
from class_balanced_loss import CB_loss
from Utils import adjust_learning_rate

class TrainerWrapper:
    def __init__(self,configParser):
        self.configParser = configParser
        self.initialize_variables()
        return
    def initialize_variables(self):
        self.weight_decay = self.configParser.getfloat("TRAINING","weight_decay")
        self.learning_rate = self.configParser.getfloat("TRAINING","learning_rate")
        self.loader_job = self.configParser.getint("TRAINING","loader_job")
        self.batch_size = self.configParser.getint("TRAINING","batch_size")
        self.device_id = self.configParser.get("TRAINING","device_id")
        self.epoch = self.configParser.getint("TRAINING","epoch")
        self.trigger_epoch = self.configParser.getint("TRAINING","triger_epoch")
        self.trainer_logs_out = self.configParser.get("TRAINING","trainer_logs")
        self.balanced_iterator = self.configParser.getboolean("TRAINING","balanced_iterator")
        self.batch_balancing = self.configParser.getboolean("TRAINING","batch_balancing")
        self.n_processes = self.configParser.getint("TESTING","n_processes")
        self.loss_function_name = self.configParser.get("TRAINING","loss_function")
        self.balanced_loss = self.configParser.getboolean("TRAINING","balanced_loss")
        self.beta = self.configParser.getfloat("TRAINING","beta")
        self.gamma = self.configParser.getfloat("TRAINING","gamma")
        self.loss_type = self.configParser.get("TRAINING","loss_function")
        self.checkpoint_file = self.configParser.get("DATA","checkpoint_file")

        create_directory_if_not_exists(self.trainer_logs_out)
        
    def train_model(self, model, train_data, test_data, number_of_classes, y_train, counts):
        #Do we need? Yes we dp
        # model = torch.nn.DataParallel(model)
        criterion = nn.CrossEntropyLoss().cuda()
        
        #optimizer = optim.Adadelta(model.parameters(), lr = self.learning_rate)#, weight_decay = self.weight_decay)
        #optimizer = optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        optimizer = optim.RMSprop(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        best_so_far = 0
        state = None
        for i_epoch in range(self.epoch):
            adjust_learning_rate(self.learning_rate, optimizer, i_epoch)
            train_loss, batches = self.train(model, train_data, criterion, optimizer,i_epoch, best_so_far, counts, number_of_classes)
            best_so_far, test_loss, test_batches, precision, recall, f1, accuracy, state = self.test(model, test_data, criterion, i_epoch, best_so_far, counts, number_of_classes, state)
            
            print(f"Epoch {i_epoch+1}/{self.epoch}, training loss: {train_loss/batches}, validation loss: {test_loss/test_batches}")
            self.print_scores(precision, recall, f1, accuracy, test_batches)
        if not os.path.isdir(self.trainer_logs_out + self.checkpoint_file):
            os.mkdir(self.trainer_logs_out + self.checkpoint_file)
        torch.save(state,self.trainer_logs_out + self.checkpoint_file + 'ckpt.t7')
            

        return model
    
    def train(self, model, train_data, criterion, optimizer, c_epoch, best_so_far, counts, number_of_classes):
        
        print('\nEpoch: %d' % c_epoch)
        model.cuda()
        model.train()
        
        train_loss = 0
        
        correct = 0
        total = 0
        
        train_loader = torch.utils.data.DataLoader(train_data, 
                                                   batch_size=self.batch_size, 
                                                   shuffle=True, 
                                                   num_workers=self.n_processes,
                                                   collate_fn=train_data.collate_fn)
        batches = len(train_loader)
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
        
        for batch_idx, batch in progress:
        #for batch_idx, (inputs, targets) in enumerate(train_loader, 0):
            inputs, targets, lenghts = batch['sentences'].cuda(), batch['labels'].cuda(), batch['lengths'].cuda()
            
            outputs = model(inputs,lenghts)
            if self.balanced_loss:
                loss = CB_loss(targets, outputs, counts, number_of_classes, self.loss_type, self.beta, self.gamma)
            else:
                loss = criterion(outputs, targets)
                
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=3)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            #if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
            #    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))            
        # releasing unceseccary memory in GPU
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
            
        return train_loss, batches



    def test(self, model, test_data, criterion, epoch, best_so_far, counts, number_of_classes, state=None):
        
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        precision, recall, f1, accuracy = [], [], [], []
        test_loader = torch.utils.data.DataLoader(test_data, 
                                                  batch_size=self.batch_size, 
                                                  shuffle=False, 
                                                  num_workers=self.n_processes,
                                                  collate_fn=test_data.collate_fn)
        test_batches = len(test_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs, targets, lengths = batch['sentences'].cuda(), batch['labels'].cuda(), batch['lengths'].cuda()
                outputs = model(inputs,lengths)
                if self.balanced_loss:
                    loss = CB_loss(targets, outputs, counts, number_of_classes, self.loss_type, self.beta, self.gamma)
                else:
                    loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # calculate P/R/F1/A metrics for batch
                for acc, metric in zip((precision, recall, f1, accuracy), (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        self.calculate_metric(metric, targets.cpu(), predicted.cpu())
                    )
            
            
        
        # Save checkpoint
        
        acc = 100.*correct/total
        if acc > best_so_far:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            best_so_far = acc
            
            
        return best_so_far, test_loss, test_batches, precision, recall, f1, accuracy, state
    
    
    
    def calculate_metric(self, metric_fn, true_y, pred_y):
        # multi class problems need to have averaging method
        if "average" in inspect.getfullargspec(metric_fn).args:
            return metric_fn(true_y, pred_y, average="macro")
        else:
            return metric_fn(true_y, pred_y)
    
    def print_scores(self, p, r, f1, a, batch_size):
        # just an utility printing function
        for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
            print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")