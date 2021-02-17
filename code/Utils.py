
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


# In[404]:
#this function is used for parsing the data
def parsedata(filepath,resourcepath, name):

    count=0
    anncount=0
    sentence=[]
    
    annotation=[]
    annotations=[]
    babelnet=[]
    provisore=[]
    flag=0
    logg=0
    v=0
    indexes=[]
    indexi=0
    indexj=0
    prev=[]
    senX=[]
    ids=[]
    listid=[]
    pos=[]
    POS=[]
    #iterate over xml file
    for event, elem in ET.iterparse(filepath):
        #execute on every sentence start. I upload the data extracted from file , like lemmas, senses, pos and the index position of each sentence related to the file in 
        #different files
        #i also check if the sentence length goes beyond 30 , in which case i devide it, so despite using padding we never trunkate the sentance and lose data
        if elem.tag=='sentence':
            if senX!=[]:
                if len(senX)>30:
                    for i in range(0, len(senX), 29):
                        v=[]
                        prev2=[]
                        for elemz in prev:
                            prev2.append([indexi,elemz[1]%29])
                        
                        
                        indexes.extend(prev2[i:i+29])
                        
                        indexi+=1
                        temp=senX[i:i+29]
                        sentence.append(temp)
                        
                else:
                    temp=senX
                    sentence.append(temp)
                    indexes.extend(prev)
                    indexi+=1
            
            if pos!=[]:
                if len(pos)>30:
                    for i in range(0, len(pos), 29):
                        temp2=pos[i:i+29]
                        POS.append(temp2)
                    
                
                else:
                    temp2=pos
                    POS.append(temp2) 
                         
            senX=[]
        
            pos=[]
            prev=[]
            
            indexj=0
            count+=1
            clear_output()
            print(count)
        if elem.tag=='wf':
            if elem.get('lemma').isalpha() or elem.get('lemma').isdigit():
                indexj+=1
                senX.append(elem.get('lemma'))
                pos.append(elem.get('pos'))
        if elem.tag=='instance':
            
            indexj+=1
            senX.append(elem.get('lemma'))
            pos.append(elem.get('pos'))
            prev.append([indexi,indexj-1])
            ids.append(elem.get('id'))

    return indexes, sentence, POS, ids


# In[405]:
#Function to convert the parsed data into vectors

def processdata(filearg, vocabname):

    x=filearg        

    with open(vocabname, 'rb') as handle:
            vocabs=pickle.load(handle)
    print('preprocessing start')
    x_train=[]

    for sen in x:
        w=[]
        for elem in sen:
            if elem.isdigit():
                w.append(vocabs['<SUB>'])
            elif elem not in vocabs:
                w.append(vocabs['<UNK>'])
            else:
                w.append(vocabs[elem])
        x_train.append(w)
    
    print('x finished')
 

    X_train=keras.preprocessing.sequence.pad_sequences(x_train,truncating='pre',padding='post',maxlen=30)

    return X_train, vocabs


# In[406]:
#this maps the map files into specific dictionaries
def buildmaps(resourcepath):
    babelnet=dict()
    prov=[]
    with open(resourcepath+'babelnet2wordnet.tsv', 'r') as f:
        for line in f:
            prov=line.split()
            babelnet[prov[1]]=prov[0]
    domains=dict()
    prov=[]
    with open(resourcepath+'babelnet2wndomains.tsv', 'r') as f:
        for line in f:
            prov=line.split()
            domains[prov[0]]=prov[1]
            
    lex=dict()
    prov=[]
    with open(resourcepath+'babelnet2lexnames.tsv', 'r') as f:
        for line in f:
            prov=line.split()
            lex[prov[0]]=prov[1]
    return babelnet, domains, lex


# In[407]:

def processing(datapath,resourcepath):
    indexes, x, pos, ids=parsedata(datapath,resourcepath,'test')
    xtrain, vocab=processdata(x,resourcepath+'finalvocab.pickle' )
    with open('xtest.pickle', 'wb') as handle:
        pickle.dump(xtrain, handle)
    babelnet, domains, lex=buildmaps(resourcepath)
    
    return xtrain, babelnet,domains,lex,vocab,x, pos,ids, indexes


# In[408]:
#this function is needed to fix issues with instance position index when i devide the sentences in parsing phase.
def modindexing(indexes):
    curr=0
    prov=0
    for i in range(len(indexes)-1):
        if int(indexes[i][0])==(indexes[i+1][0]): 
            if int(indexes[i][1])>=int(indexes[i+1][1]):
                indexes[i+1][0]+=1
           
        if int(indexes[i][0])>(indexes[i+1][0]):
            indexes[i+1][0]=indexes[i][0]  
            
    print('done')       
    return indexes


# In[409]:
#the 3 below functions make sure to convert the extracted data with senses into babelnet senses, domains or lexes
def ybabel(y,babelnet):
    yList=[]
    indexi=0
    for row in y:
        ylist=[]
        indexi+=1
        indexj=0
        for word in row:
            indexj+=1
            if 'wn:' not in word:
                ylist.append(word)
            else:
                ylist.append(babelnet[word])
        yList.append(ylist)
    return yList


def ylex(y, lex):
    yList=[]
    indexi=0
    for row in y:
        ylist=[]
        indexi+=1
        indexj=0
        for word in row:
            indexj+=1
            if 'bn:' not in word:
                ylist.append(word)
            else:
                ylist.append(lex[word])
        yList.append(ylist)
    return yList  

def ydomain(y, domains):
    yList=[]
    occ=dict()
    index=[]
    indexi=0
    for row in y:
        ylist=[]
        indexi+=1
        indexj=0
        for word in row:
            indexj+=1
            if 'bn:' not in word:
                ylist.append(word)
            else:
                if word in domains:
                    ylist.append(domains[word])
                    if domains[word] not in occ:
                        
                        occ[domains[word]]=1
                    else:
                        occ[domains[word]]+=1
                else:
                    index.append([indexi-1,indexj-1])
                    ylist.append(word)
        yList.append(ylist)
    
    if len(occ)!=0:    
        saveword=max(occ.items(), key=operator.itemgetter(1))[0]
    
        print('time to replace')
        for elem in index:
            yList[elem[0]][elem[1]]=saveword
    else:
        for elem in index:
            yList[elem[0]][elem[1]]='factotum'
    return yList  
