
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
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
import pickle
import keras as K
import tensorflow
from nltk.corpus import wordnet as wn
import xml.etree.ElementTree as ET
import keras
import operator
from Model import buildmodel


#function that calls the model
model=buildmodel()


# In[ ]:

#This is the generator class. It consists of series of functions that are called at the beginning and during the training
#The way it works is it takes file paths as arguments and a list of sentence lengths. It opens the files, uses the seek() function to 
#extract the sentences using aswell the sentence lengths and it pads it , adds it into a vector with batch size entries and returns the batch
class DataGeneratorMulti(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_train, y_train,y_train2,y_train3, batch_size,
                 n_classes,length, lens,shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.Ydata1 = y_train
        self.Ydata2 = y_train2
        self.Ydata3 = y_train3
        self.Traindata = x_train
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.length=length
        self.on_epoch_end()
        self.lens=lens

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.length / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        

        list_Ttemp=[]
        list_Ytemp1=[]
        list_Ytemp2=[]
        list_Ytemp3=[]
        #extract sentences
        for i in indexes:
                with open(self.Traindata, 'r') as handle:
                    if i==0:
                        list_Ttemp.append(np.asarray(list(map(int,handle.readline().split()))))
                    else:
                        handle.seek(0.0)
                        handle.seek(lens[0][i-1])

                        list_Ttemp.append(np.asarray(list(map(int,handle.readline().split()))))

        
       
            
                with open(self.Ydata1, 'r') as handle1:
                    if i==0:
                        list_Ytemp1.append(np.asarray(list(map(int,handle1.readline().split()))))
                    else:
                        handle1.seek(0.0)
                        handle1.seek(lens[1][i-1])
                        list_Ytemp1.append(np.asarray(list(map(int,handle1.readline().split()))))

        
        
        
                with open(self.Ydata2, 'r') as handle2:
                    if i==0:
                        list_Ytemp2.append(np.asarray(list(map(int,handle2.readline().split()))))
                    else:
                        handle2.seek(0.0)
                        handle2.seek(lens[2][i-1])

                        list_Ytemp2.append(np.asarray(list(map(int,handle2.readline().split()))))


    
        
                with open(self.Ydata3, 'r') as handle3:
                    if i==0:
                        list_Ytemp3.append(np.asarray(list(map(int,handle3.readline().split()))))
                    else:
                        handle3.seek(0.0)
                        handle3.seek(lens[3][i-1])

                        list_Ytemp3.append(np.asarray(list(map(int,handle3.readline().split()))))

        
        
        ttemp=keras.preprocessing.sequence.pad_sequences(list_Ttemp,truncating='pre',padding='post',maxlen=30 )
        ytemp=keras.preprocessing.sequence.pad_sequences(list_Ytemp1,truncating='pre',padding='post',maxlen=30 )
        ytemp=ytemp.reshape(*(ytemp).shape,1)
   

        ytemp2=keras.preprocessing.sequence.pad_sequences(list_Ytemp2,truncating='pre',padding='post',maxlen=30 )
        ytemp2=ytemp2.reshape(*(ytemp2).shape,1)
        
        ytemp3=keras.preprocessing.sequence.pad_sequences(list_Ytemp3,truncating='pre',padding='post',maxlen=30 )
        ytemp3=ytemp3.reshape(*(ytemp3).shape,1)
        return ttemp, [ytemp,ytemp2,ytemp3]
      

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.length)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)




# In[ ]:
#gets the list that holds the lengths of sentences for both train and eval datasets
with open('indexes2.pickle', 'rb') as handle:
    lens=pickle.load(handle)

with open('indexesval.pickle', 'rb') as handle:
    lensval=pickle.load(handle)


# Generators
training_generator = DataGeneratorMulti('x_traingen.txt', 'y_traingen.txt','y_traindomaingen.txt','y_trainlexgen.txt',batch_size,40102,42838,lens,True)

validation_generator=DataGeneratorMulti('x_valgen.txt', 'y_valgen.txt','y_valdomaingen.txt','y_vallexgen.txt', batch_size, 15879,151,lensval, True)

# In[ ]:

#############################################################
###########  Train part ##############################
###############################################
cbk = K.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')
print("\nStarting training...")

model.fit_generator(generator=training_generator,validation_data=validation_generator,epochs=epochs,use_multiprocessing=True,
                    workers=8,callbacks=[cbk])
print("Training complete.\n")

#save the model configuration and weights
model.save('my_modeltest.h5')
model.save_weights('modelweightstestsplit.h5')

