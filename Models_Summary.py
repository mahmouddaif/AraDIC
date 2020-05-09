#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:27:23 2019

@author: daif
"""
from Models.CharacterEncoder import CharacterEncoder
from torchsummary import summary
from ConfigParser import ConfigsLoader
from Models.DeepCLCNNwithWL import  DeepCLCNNwithWL
from Models.DeepCLCNNNoEncoder import DeepCLCNNNoEncoder
from Models.DeepNoPoolingCLCNNNoEncoder import DeepNoPoolingCLCNNNoEncoder
from Models.DeepNoPoolingCLCNN import DeepNoPoolingCLCNN
from Models.DeepNoPoolingCLCNNwithWT import DeepNoPoolingCLCNNwithWT
from Models.PositionalEncoder import PositionalEncoder
from Models.CNNSentence import CNNSentence
from Models.RNNCharacter import RNNCharacter
from Models.RNNImageEncoder import RNNImageEncoder

def model_summary(model):
  print("model_summary")
  print()
  print("Layer_name"+"\t"*7+"Number of Parameters")
  print("="*100)
  model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
  layer_name = [child for child in model.children()]
  j = 0
  total_params = 0
  print("\t"*10)
  for i in layer_name:
    print()
    param = 0
    try:
      bias = (i.bias is not None)
    except:
      bias = False  
    if not bias:
      param =model_parameters[j].numel()+model_parameters[j+1].numel()
      j = j+2
    else:
      param =model_parameters[j].numel()
      j = j+1
    print(str(i)+"\t"*3+str(param))
    total_params+=param
  print("="*100)
  print(f"Total Params:{total_params}")       


def get_character_encoder_summary():
    configLoader = ConfigsLoader("/media/daif/01D447B0862F25C01/Hosei University/Projects/Arabic-Document-Classification/Deep_Learning/PyTorch/Configs.ini")
    configs = configLoader.get_configs()
    encoder = CharacterEncoder(configs,64)
    print(summary(encoder, (1,36,36)))
    model_summary(encoder)

def get_DeepCLCNNwithWL_summary():
    configLoader = ConfigsLoader("/media/daif/01D447B0862F25C01/Hosei University/Projects/Arabic-Document-Classification/Deep_Learning/PyTorch/Configs.ini")
    configs = configLoader.get_configs()
    model = DeepCLCNNwithWL(configs,10)
    print(summary(model, (60,36,36)))

def get_DeepCLCNNNoEncoder_summary():
    configLoader = ConfigsLoader("/media/daif/01D447B0862F25C01/Hosei University/Projects/Arabic_Document_Classification_Pytorch/Configs.ini")
    configs = configLoader.get_configs()
    model = DeepCLCNNNoEncoder(configs,10)
    print(summary(model, (227,128)))


def get_DeepNoPoolingCLCNNNoEncoder_summary():
    configLoader = ConfigsLoader("/media/daif/01D447B0862F25C01/Hosei University/Projects/Arabic_Document_Classification_Pytorch/Configs.ini")
    configs = configLoader.get_configs()
    model = DeepNoPoolingCLCNNNoEncoder(configs,10)
    print(summary(model, (227,128)))

def get_DeepNoPoolingCLCNN_summary():
    configLoader = ConfigsLoader("/media/daif/01D447B0862F25C01/Hosei University/Projects/Arabic_Document_Classification_Pytorch/Configs.ini")
    configs = configLoader.get_configs()
    model = DeepNoPoolingCLCNN(configs,10)
    print(summary(model, (128,36,36)))

def get_DeepNoPoolingCLCNNwithWT_summary():
    configLoader = ConfigsLoader("/media/daif/01D447B0862F25C01/Hosei University/Projects/Arabic_Document_Classification_Pytorch/Configs.ini")
    configs = configLoader.get_configs()
    model = DeepNoPoolingCLCNNwithWT(configs,10)
    print(summary(model, (128,36,36)))

def get_PositionalEncodingSummary():
    model = PositionalEncoder(128,60)
    print(summary(model, (128,60)))
    
def get_CNNSentenceSummary():
    configLoader = ConfigsLoader("/media/daif/01D447B0862F25C01/Hosei University/Projects/Arabic-Document-Classification/Deep_Learning/PyTorch/Configs.ini")
    configs = configLoader.get_configs()
    model = CNNSentence(configs,10)
    print(summary(model, (255, 111000)))
    
def get_RNNSentenceSummary():
    configLoader = ConfigsLoader("/media/daif/01D447B0862F25C01/Hosei University/Projects/Arabic-Document-Classification/Deep_Learning/PyTorch/Configs.ini")
    configs = configLoader.get_configs()
    model = RNNCharacter(configs,10)
    print(summary(model, (60,36,36)))

def get_RNNImageEncoder():
    configLoader = ConfigsLoader("/media/daif/01D447B0862F25C01/Hosei University/Projects/Arabic-Document-Classification/Deep_Learning/PyTorch/Configs.ini")
    configs = configLoader.get_configs()
    model = RNNImageEncoder(configs)
    print(summary(model, (10,10)))

#get_PositionalEncodingSummary()
get_RNNImageEncoder()