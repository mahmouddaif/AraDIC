#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 07:34:38 2020

@author: daif
"""

from torch.utils.data.dataset import Dataset


from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)

import numpy as np
import arabic_reshaper
from Clean_Arabic_Text import remove_none_arabic_characters
import logging
import torch

class TextDatasetCharToImage(Dataset):
    def __init__(self,configParser, sentences, labels,chars_df,**kwargs):
        self.configParser = configParser
        self.initialize_variables()
        self.sentences = sentences
        self.labels = labels
        self.load_font()
        #self.initialize_loggeer()
        self.chars_df = chars_df
        self.initialize_char_image_dict()
        self.hits = 0
        self.misses = 0
        self.number_of_basic_sentences = 0
    
    def initialize_loggeer(self):
        f_handler = logging.FileHandler(self.logging_file)
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        self.logger = logging.getLogger()  
        self.logger.addHandler(f_handler)
        self.logger.info("TextDataset __init__ end")
    
    
    def initialize_variables(self):
        self.image_width = self.configParser.getint("TEXT_TRANSFORMATION","image_width")
        self.image_height = self.configParser.getint("TEXT_TRANSFORMATION","image_height")
        self.font_size = self.configParser.getint("TEXT_TRANSFORMATION","font_size")
        self.fntFile = self.configParser.get("DATA","font_file")
        self.chars_file_name = self.configParser.get("DATA","chars_csv_file_name")
        self.char_df_colNames = [self.configParser.get("DATA","char_df_column")]
        self.sentence_start = self.configParser.getint("TEXT_TRANSFORMATION","start")
        self.sentence_end = self.configParser.getint("TEXT_TRANSFORMATION","end")
        self.logging_file = self.configParser.get("DATA","log_file")

    def initialize_char_image_dict(self):

        self.char_image_dict = {}
        for index, row in self.chars_df.iterrows():
            character = str(row["Char"]).strip()
            character_image = self.draw_image(character, self.image_width, self.image_height, self.fnt)
            if not(character in self.char_image_dict):
                self.char_image_dict[character] = character_image
        self.char_image_dict[None] = self.draw_image(None, self.image_width, self.image_height, self.fnt)
        return self.char_image_dict
    def load_font(self):
        
        self.fnt = ImageFont.truetype(self.fntFile, int(self.font_size * 0.85), encoding="utf-8")
        
    def string_to_image_matrxi(self, string):
        charsImages = []
        maxChars = min(self.sentence_end,len(string))
        subStr =  string[self.sentence_start:(self.sentence_start+maxChars)]
        #The image width and height
        
        for char in subStr:
            char = str(char)
            img = None
            #print("The char is: ", char)
            if char in self.char_image_dict:
                self. hits+=1
                img = self.char_image_dict[char]
            else:
                self.misses+=1
                img = self.draw_image(char,self.image_width, self.image_height, self.fnt )
                self.char_image_dict[char] = img
            #img = img*1.0
            img = img.astype(np.float32)
            charsImages.append(img)
            #print(img.dtype)
        if len(charsImages) < self.sentence_end:
            restChars = self.sentence_end - len(charsImages)
            for i in range(restChars):
                img = self.char_image_dict[None]
                #img = img*1.0
                img = img.astype(np.float32)
                charsImages.append(img)
        charsImages = np.asarray(charsImages)
        charsImages = charsImages.astype(np.float32)
        return charsImages
    
    def draw_image(self, char, W, H, fnt):
        img = Image.new('L', (W, H),  "black")
        #print(img)
        d = ImageDraw.Draw(img)
        #print(char)
        if char is not None:
            w, h = d.textsize(char, font=fnt)
            d.text(((W-w)/2,(H-h)/2), str(char), font=fnt, fill="#fff")
        img = np.asarray(img)
        #print(img.shape)
        #print(img.reshape(img.shape[0]*img.shape[1]))
        #plt.imshow(img, cmap='Greys')
        img = img * (1.0 / 255.0)
        return img
    
    def __getitem__(self, i):
        sentence, label = self.sentences[i], self.labels[i]
        return (sentence,label)
        
    def __len__(self):
        return len(self.sentences)
    
    def collate_fn(self, data):
        
        sentences = list()
        labels = list()
        lengths = list()
        
        for item in data:
            sentence = item[0]
            length = len(sentence)
            sentence = remove_none_arabic_characters(sentence)
            sentence = self.string_to_image_matrxi(sentence)
            sentence = torch.FloatTensor(sentence)
            label = item[1]
            sentences.append(sentence)
            labels.append(label)
            lengths.append(length)
            
        labels = torch.LongTensor(labels)
        sentences = torch.stack(sentences)
        lengths = torch.LongTensor(lengths)
            
        return {
            'sentences': sentences,
            'labels': labels,
            'lengths': lengths
        }


