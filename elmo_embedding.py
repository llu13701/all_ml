#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:23:08 2019

@author: louisalu

https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/
<<<<<<< HEAD
https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
=======

>>>>>>> 1ace022d5b99f1fc69d8ae85e0426d9e8b351818

"""

#import os
#os.chdir('/Users/louisalu/Documents/local_twilio/git_twilio')

import pandas as pd
import numpy as np
import spacy
import pickle
from numba import cuda

pd.set_option('display.max_colwidth', 200)

import tensorflow as tf
import tensorflow_hub as hub

tf.compat.v1.disable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
nlp = spacy.load('en', disable=['parser', 'ner'])
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
 
#elmo for text
#train=pd_text['text']
pd_text_list_new=pd.read_csv("question_answer_corpus_pair.csv", index_col=0)
pd_text_list_new.columns=['text', 'classification']
print ("downloaded the files")

full_corpus=pd_text_list_new['text']

#train=full_corpus.iloc[0:487438]
train=full_corpus.iloc[487438:]

config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True

# Change these two only
counter=0
filepart=8

list_train = [train[i:i+100] for i in range(counter,train.shape[0],100)]
#list_train = [train[i:i+100] for i in range(counter,300,100)]

print ("start testing elmo")

x=tf.compat.v1.placeholder("string", None)
embeddings = tf.reduce_mean(elmo(x, signature="default", as_dict=True)["elmo"], 1)

with tf.compat.v1.Session(config=config) as sess:
    result_list=[]
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.tables_initializer())
    for i in range(counter, len(list_train)):
        print ("running " , i)
        result = sess.run(embeddings, feed_dict={x:list_train[i].tolist()})
        result_list.append(result)

final_output=np.concatenate(result_list, axis = 0)

pickle_out = open("elmo_test_11022019_part.pickle", 'wb')
pickle.dump(final_output, pickle_out)
pickle_out.close()


'''
# Change these two only
counter_OutSample=0
filepart_OutSample=1

list_test = [test_sample[i:i+100] for i in range(counter_OutSample,test_sample.shape[0],100)]
print ("start training test elmo")

elmo_train=[]
for x in list_test:
    counter_OutSample=counter_OutSample+1
    print ("out of sample testing round ", counter_OutSample)
    elmo_train.append(elmo_vectors(x, config))
    if counter_OutSample%50==0:
      print ("Writing to pickle file and then reassigning elmo_train to []")
      temp=np.concatenate(elmo_train, axis = 0)
      pickle_out = open("elmo_test_10272019_part%d.pickle" %filepart_OutSample ,"wb")
      pickle.dump(temp, pickle_out)
      pickle_out.close()
      elmo_train=[]
      filepart_OutSample=filepart_OutSample+1

#elmo_train = [elmo_vectors(x) for x in list_test]
print ("finish training test elmo")

'''