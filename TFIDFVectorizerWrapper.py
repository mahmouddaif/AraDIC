#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:36:36 2020

@author: daif
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

class TFIDFVectorizerWrapper:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        return
    def get_tf_idf_df(self, documents):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        df = pd.DataFrame(tfidf_matrix.toarray(), columns = self.tfidf_vectorizer.get_feature_names())
        return df
    
    
def test_tf_idf_vectorizer():
    documents = [
            "احا الشبشب ضاع",
            "احا ده كان بصباع",
            "احا ده لسة جديد",
            "احا ده كان للعيد"
            ]
    tfidf_vect = TFIDFVectorizerWrapper()
    tf_idf_df = tfidf_vect.get_tf_idf_df(documents)
    print(tf_idf_df["احا"].values)
    list1 = [1, 2, 3, 4, 5, 6]
     
    # Convert list1 into a NumPy array
    a1 = np.array(list1)
    print(a1)

    
#test_tf_idf_vectorizer()

        
    
    