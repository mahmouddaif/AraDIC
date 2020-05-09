#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:24:46 2018

@author: daif
"""
import os
import pandas as pd
import numpy as np

from ConfigParser import ConfigsLoader
import datetime
import time
import sys
import torch

def load_model(model_file, model):
    print("Now loadinng the model ")
    loaded_model = torch.load(model_file)
    print("Accuracy, ", loaded_model["acc"])
    print("epoch, ", loaded_model["epoch"])
    model.load_state_dict(loaded_model["net"])
    return model

def load_chars_df(file_name):
    chars_df = pd.read_csv(file_name, names=["Char"])
    return chars_df

def test_load_chars_df():
    file_name = 'data/Arabic_Letters.csv'
    chars_df = load_chars_df(file_name)
    print(chars_df.columns.values)
    print(chars_df.dtypes)
    print(len(chars_df))
    print(chars_df)
    return

def write_numpy_array_to_file(arr, file_name):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_name, 'w') as f:
        for image in arr:
            f.write(np.array2string(image, separator=', '))
            f.write("\n")
    return

def load_csv_dataset(fileName, sep, colNames):
    data = pd.read_csv(fileName, sep=sep, names=colNames)
    return data

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return

def test_load_csv_dataset():
    configFileName = "Configs.ini"
    configLoader = ConfigsLoader(configFileName)
    configs = configLoader.get_configs()
    fileName = configs.get("DATA","csv_file_name")
    colNames = configs.get("DATA","col_names")
    data = load_csv_dataset(fileName, "\t", colNames)
    data = data[1:11]
    print("Number of records: ",len(data))
    print("Dataset columns: ", data. columns.values)
    print(data.head())

def test_load_chars_dataset():
    configFileName = "Configs.ini"
    configLoader = ConfigsLoader(configFileName)
    configs = configLoader.get_configs()
    fileName = configs.get("DATA","chars_csv_file_name")
    colNames = [configs.get("DATA","char_df_column")]
    data = load_csv_dataset(fileName, ",", colNames)
    data = data[1:11]
    print("Number of records: ",len(data))
    print("Dataset columns: ", data. columns.values)
    print(data.head())
def rep_sample(df, col, n, *args, **kwargs):
    nu = df[col].nunique()
    mpb = n // nu
    mku = n - mpb * nu
    fills = np.zeros(nu)
    fills[:mku] = 1

    sample_sizes = (np.ones(nu) * mpb + fills).astype(int)

    gb = df.groupby(col)

    sample = lambda sub_df, i: sub_df.sample(sample_sizes[i], *args, **kwargs)

    subs = [sample(sub_df, i) for i, (_, sub_df) in enumerate(gb)]

    return pd.concat(subs)

def construct_count_array(t):
    unique, counts = np.unique(t,return_counts=True)
    counts = np.asarray(counts)
    #print(type(counts))
    return counts

def test_construct_count_array():
    t = np.array([1, 0]).astype(np.int32)
    #print(x.shape[1])
    counts = construct_count_array(t)
    print(counts)

def construct_embedding_random_noise_array(std, size):
    noise = np.random.normal(0,1,100)
    return noise

def test_random_noise_array():
    std = 10.0
    size = 20 
    noise = np.random.normal(0,std,size)
    print(noise)

    
def construct_df_from_dict(dictionary):
    df = pd.DataFrame.from_dict([dictionary])
    return df

def get_current_datetime_string():
    now = datetime.datetime.now()
    now = now.strftime("%B %d, %Y %H:%M:%S")
    return now

def append_two_dataframes(df1, df2):
    df1 = df1.append(df2)
    return df1
"""
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
"""
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 8 epochs"""
    lr = lr * (0.1 ** (epoch // 8))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



