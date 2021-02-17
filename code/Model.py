
# coding: utf-8

# In[1]:

import math
from tqdm import tqdm
import networkx as nx
import numpy as np
import os
from IPython.display import clear_output
from collections import Counter, namedtuple
import multiprocessing
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, FastText
import pickle
import keras as K
import tensorflow
from nltk.corpus import wordnet as wn
import xml.etree.ElementTree as ET
import keras
import operator


# In[79]:

 #this is a custom layer used to remove the masking and fix the mask support problem for the flatten() layer
class NonMasking(keras.layers.Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
  
    def build(self, input_shape):   
        input_shape = input_shape   
  
    def compute_mask(self, input, input_mask=None):   
        # do not pass the mask to the next layers   
        return None   
  
    def call(self, x, mask=None):   
        return x   
  
    def get_output_shape_for(self, input_shape):   
        return input_shape
    
#model function
def mymodel(vocab_size,vocab_size2,vocab_size3,embedding_size, hidden_size, epochs):
    print("Model is starting")
    model=K.models.Sequential()
    
    #Embedding layer
   
    x=K.layers.Input(shape=(30,))
    X_x=K.layers.Embedding(vocab_size,128,mask_zero=True)(x)
    #LSTM layer
    lstm=(K.layers.Bidirectional(K.layers.LSTM(hidden_size,dropout=0.2, recurrent_dropout=0.3,return_sequences=True)))(X_x)
   
    #attention goes here
    
    attention = K.layers.TimeDistributed(K.layers.Dense(1, activation='tanh'))(lstm)
    attention=NonMasking()(attention)
    attention = K.layers.Flatten()(attention)
    attention = K.layers.Activation('softmax')(attention)
    attention = K.layers.RepeatVector(128)(attention)
    attention = K.layers.Permute([2, 1])(attention)

    sent_representation = K.layers.multiply([lstm, attention])
    lstm2 = keras.layers.Lambda(lambda xin: np.sum(xin, axis=0))(sent_representation)

    #dense layer
    lstm2 = K.layers.Masking(mask_value=0)(lstm2)
    y1=K.layers.TimeDistributed(K.layers.Dense(vocab_size,activation="softmax",name="senses"))(lstm2)
    y2=K.layers.TimeDistributed(K.layers.Dense(vocab_size2,activation="softmax",name="domains"))(lstm2)          
    y3=K.layers.TimeDistributed(K.layers.Dense(vocab_size3,activation="softmax",name="lex"))(lstm2)          
 
    model=K.models.Model(inputs=[x],outputs=[y1,y2,y3])
    
    #Optimizer
    optimizer=K.optimizers.Adam(lr=0.02,decay=0.001/epochs,amsgrad=False)
              
    #Compile
    model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def buildmodel():
    # In[80]:

    vocab_size=41012
    vocab_size2=41012
    vocab_size3=41012
    HIDDEN_SIZE = 64
    batch_size = 64
    epochs = 10

    model=mymodel(vocab_size,vocab_size2,vocab_size3,vocab_size,HIDDEN_SIZE,epochs)
    return model
