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

import os
os.chdir('/Users/louisalu/Documents/twilio-sms/git_twilio')

import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
pd.set_option('display.max_colwidth', 200)
from sklearn.model_selection import train_test_split
import random

nlp = spacy.load('en', disable=['parser', 'ner'])

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
 
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output


def convert_question_to_binary(text):
    if 'question' in text.lower():
        return 1
    else:
        return 0


pd_text=pd.read_csv("nltk_chat_cleaned.csv", index_col = 0)
pd_text.columns=['text', 'type']

binary=[convert_question_to_binary(x) for x in pd_text['type']]
pd_text['binary_classification']=binary

#REMOVE links and all
pd_text['text']=pd_text['text'].apply(lambda x: re.sub(r'http\S+', '', x))
pd_text['text']=pd_text['text'].apply(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', x))
pd_text['text'] = pd_text['text'].str.lower()
pd_text['text']=pd_text['text'].str.replace("[0-9]", " ")
pd_text['text'] = pd_text['text'].apply(lambda x:' '.join(x.split()))
pd_text['text']=lemmatization(pd_text['text'])

del pd_text['type']  
#10567

pd_quora=pd.read_csv("train.csv", lineterminator='\r',error_bad_lines=False)
pd_quora.columns=['text']
pd_quora['text']=pd_quora['text'].apply(lambda x: re.sub(r'http\S+', '', str(x)))
pd_quora['text']=pd_quora['text'].apply(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', str(x)))
pd_quora['text'] = pd_quora['text'].str.lower()
pd_quora['text']=pd_quora['text'].str.replace("[0-9]", " ")
pd_quora['text'] = pd_quora['text'].apply(lambda x:' '.join(x.split()))
pd_quora['text']=lemmatization(pd_quora['text'])

binary_one=[1 for x in pd_quora['text']]
pd_quora['binary_classification']=binary_one
#404290


pd_twits=pd.read_csv("twitter_statement.csv", lineterminator='\r',error_bad_lines=False)
pd_twits.columns=['text']
pd_twits['text']=pd_twits['text'].apply(lambda x: re.sub(r'http\S+', '', str(x)))
pd_twits['text']=pd_twits['text'].apply(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', str(x)))
pd_twits['text'] = pd_twits['text'].str.lower()
pd_twits['text']=pd_twits['text'].str.replace("[0-9]", " ")
pd_twits['text'] = pd_twits['text'].apply(lambda x:' '.join(x.split()))
pd_twits['text']=lemmatization(pd_twits['text'])
binary_one=[0 for x in pd_twits['text']]
pd_twits['binary_classification']=binary_one

pd_text_combo=pd_text.append(pd_twits, ignore_index=True)
pd_text_combo=pd_text_combo.append(pd_quora, ignore_index=True)
pd_text_list=pd_text_combo.values.tolist()

pd_text_list_new=[x for x in pd_text_list if x[0]==x[0]]
pd_text_list_new=[x for x in pd_text_list_new if len(x[0])>2]
pd_text_list_new=[x for x in pd_text_list_new if len(x[0].split())>2]

pd_text_list_data=pd.DataFrame(pd_text_list_new)
pd_text_list_data.to_csv("question_answer_corpus_pair.csv")

############PREPARING FOR SAMPLE TESTING###########
random.shuffle(pd_text_list_new)
train=pd_text_list_new[0:487438]
test=pd_text_list_new[487438:]
train_x=[x[0] for x in train]
train_y=[x[1] for x in train]
test_x=[x[0] for x in test]
test_y=[x[1] for x in test]

################## TFIDF############
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer(min_df=5)
X_train_counts = count_vect.fit_transform(train_x)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


####################Using Multinomia NB###########
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
 ])

text_clf.fit(train_x, train_y)  

predicted = text_clf.predict(test_x)
np.mean(predicted == test_y)  ###94.55


text_clf_SVC = Pipeline([
    ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
 ])

text_clf_SVC.fit(train_x, train_y)  
predicted_svc = text_clf_SVC.predict(test_x)
np.mean(predicted_svc == test_y)  ###96.1

pickle_out = open("svm_classification.pickle","wb")
pickle.dump(text_clf_SVC, pickle_out)
pickle_out.close()
# load elmo_train_new
pickle_in = open("svm_classification.pickle", "rb")
classification_question = pickle.load(pickle_in)

prediction=classification_question.predict(test_x)

##################GRID SEARCH#########
parameters = {
     'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
     'clf__alpha': (1e-2, 1e-3),
 }

gs_clf = GridSearchCV(text_clf_SVC, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf = gs_clf.fit(train_x[:2000], train_y[:2000])
gs_clf.best_params_
gs_clf.predict(test_x)

