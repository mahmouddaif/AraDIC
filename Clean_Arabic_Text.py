#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:07:20 2019

@author: daif
"""
# coding=utf-8

import re
import string
from nltk.stem.isri import ISRIStemmer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations
stemmer = ISRIStemmer()

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

#To be appllied to each row of dataframe

def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text

def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)


def stem_arabic_text(text):
    text = stemmer.stem(text)
    return text

def test_arabic_stemmer():
    sentence = "طَرَقَتْكَ زَيْنَبُ بَعْدَمَا طال الكَرَى"
    print("Text before stemming is: ")
    print(sentence)
    stemmed_text = stem_arabic_text(sentence)
    print("Text after stemming is: ")
    print(stemmed_text)
    print("Now tokenizing first:")
    tokens = sentence.split()
    for token in tokens:
        print(token)
        print(stem_arabic_text(token))
    return stemmed_text

def test_remove_diacritics():
    sentence = "طَرَقَتْكَ زَيْنَبُ بَعْدَمَا طال الكَرَى"
    print("Sentence before removing diacritics: ")
    print(sentence)
    sentence_no_diacritics = remove_diacritics(sentence)
    print("Sentence after removing diacritics: ")
    print(sentence_no_diacritics)

def test_normalize_arabic():
    sentence = "وأتى المَشيبُ فحالَ دونَ شَبابي"
    print("Sentence before normalization: ")
    print(sentence)
    normalized_sentence = normalize_arabic(sentence)
    print("Sentence after normalization: ")
    print(normalized_sentence)
    



def wm2df(wm, feat_names):
    
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names)
    return(df)

def my_tokenizer(text):
    tokens = text.split()
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stem_arabic_text(token))
    return stemmed_tokens
    
def my_preprocessor(row, sentence_column):
    text = row[sentence_column]
    text = remove_diacritics(text)
    text = remove_punctuations(text)
    #text = stem_arabic_text(text)
    text = normalize_arabic(text)
    return text
    
def test_custom_analyzer():
    corpora = [
    "طَرَقَتْكَ زَيْنَبُ بَعْدَمَا طال الكَرَى",
      "وأتى المَشيبُ فحالَ دونَ شَبابي"
    ]
    
    custom_vec = CountVectorizer(preprocessor=my_preprocessor, tokenizer=my_tokenizer)
    cwm = custom_vec.fit_transform(corpora)
    tokens = custom_vec.get_feature_names()
    print("Tokens:")
    print(tokens)
    data = wm2df(cwm, tokens)
    return data


def remove_none_arabic_characters(text):
    text = re.sub('[a-zA-Z]+',  ' ',    text)
    text = text.strip()
    return text

def test_remove_none_arabic_characters():
    text = "يا ابنَ فَهْدٍو أنتَ بدرُ تَمامِ وَ حَياً صوبُه حياة ُ الأنامِ لَحَظَتْ عَزْمَتِي العِراقَفَسلَّتْ هِمَّتي للرَّحيلِ سيفَ اعتِزامِ فسَلامٌ على جَنابِكَ والمَن هَلِ والظِّلِّ والأيادي الجِسامِ غيرَ أني أريدُ منك كِتاباً مُفْرَداً يحتوي فريدَ الكَلامِ و نِظامٌ فيه الحَلالُ من السِّحْ رِ تَعالَى عن كلِّ سِحْرٍ حَرامِ يَغْتَدي منه سَمْعُ كلِّ لبيبٍ في استماعٍو قلبُه في ابتِسامِ فيه من ظاهرِ العِنايَة ِ ما يُو جِبُ حَقِّي على الأميرِ الهُمامِ فاقضِ حَقِّي فيه بساعدِ فِكْرٍ تُحْيِ شُكْري بها مَدَى الأيَّامِqtmSI"
    text = remove_none_arabic_characters(text)
    print(text)
    
#test_arabic_stemmer()

#test_remove_diacritics()

#test_normalize_arabic()
#Start from here: https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af
    
#data = test_custom_analyzer()
#print(data)
    
#test_remove_none_arabic_characters()