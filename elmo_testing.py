
#import os
#os.chdir('/Users/louisalu/Documents/local_twilio/git_twilio')

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import SGDClassifier
import keras

pd.set_option('display.max_colwidth', 200)

import tensorflow as tf

tf.compat.v1.enable_eager_execution()

import os
os.chdir('/Users/louisalu/Documents/local_twilio/git_twilio/ELMO')

pd_text_list_new=pd.read_csv("question_answer_corpus_pair.csv", index_col=0)
pd_text_list_new.columns=['text', 'classification']


# loading pretrained_dataset
pickle_in = open("elmo_train_11022019.pickle", "rb")
elmo_train = pickle.load(pickle_in)
pickle_out=open("elmo_test_11022019.pickle", "rb")
elmo_test = pickle.load(pickle_out)

cut_off_point=487438
ytrain=pd_text_list_new['classification'].tolist()[0:cut_off_point]
yvalid=pd_text_list_new['classification'].tolist()[cut_off_point:]
xtrain=elmo_train
xvalid=elmo_test

############################# TO RUN WITH SGD############
clf = SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)
clf.fit(xtrain, ytrain)  
predicted_svc = clf.predict(xvalid)

np.mean(predicted_svc == yvalid)  ###96.1

pickle_out = open("svm_elmo_classification.pickle","wb")
pickle.dump(clf, pickle_out)
pickle_out.close()

# load elmo_train_new
pickle_in = open("svm_elmo_classification.pickle", "rb")
elmo_model = pickle.load(pickle_in)


#######################Using keras deep learning ##########
ytrain_arr=np.asarray(ytrain)
yvalid_arr=np.asarray(yvalid)

model=keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
        ])
    
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(elmo_train, ytrain_arr, epochs=10)
test_loss, test_acc = model.evaluate(elmo_test,  yvalid_arr, verbose=2)
model.save('elmo_qa.h5')

