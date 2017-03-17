# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:03:17 2016

@author: Vaibhav
"""
import pickle as pick
from sklearn.feature_extraction.text import TfidfVectorizer

word_data = pick.load(open('your_word_data.pkl','r'))
print word_data[0]
vectorizer = TfidfVectorizer(stop_words='english')
word_data_transformed = vectorizer.fit_transform(word_data)

print len(vectorizer.get_feature_names())
print vectorizer.get_feature_names()[34597]
print vectorizer.vocabulary_.get('zone')