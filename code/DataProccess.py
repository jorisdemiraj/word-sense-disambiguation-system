
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


# In[2]:
#this file works the same as the Utils file, mirroring its functions
#Main difference is that every function works on extracting the Y data aswell, usable only in train and evaluation phase
#in this we do not need to save instance position as we can get the positon through the y data easily

def parsedata(filepath, goldpath, name):
    provis=[]
    mapping=dict()
    with open(goldpath, 'r') as f:
        for line in f:
            provis=line.split()
            mapping[provis[0]]=provis[1:] 
        

    count=0
    anncount=0
    sentence=[]
    Y=[]
    annotation=[]
    annotations=[]
    babelnet=[]
    provisore=[]
    flag=0
    logg=0
    v=0

    senX=[]
    senY=[]
    pos=[]
    POS=[]
    
    for event, elem in ET.iterparse(filepath):

        if elem.tag=='sentence':
            if senX!=[]:
                if len(senX)>30:
                    for i in range(0, len(senX), 29):
                    
                        temp=' '.join(senX[i:i+29])
                        sentence.append(temp)
                        
                else:
                    temp=' '.join(senX)
                    sentence.append(temp) 
            if senY!=[]:
                if len(senY)>30:
                    for i in range(0, len(senY), 29):
                        temp2=' '.join(senY[i:i+29])
                        Y.append(temp2)
                   
                else:
                    temp2=' '.join(senY)
                    Y.append(temp2) 
            if pos!=[]:
                if len(pos)>30:
                    for i in range(0, len(pos), 29):
                        temp3=' '.join(pos[i:i+29])
                        POS.append(temp3)
                    
                
                else:
                    temp3=' '.join(pos)
                    POS.append(temp3) 
                         
            senX=[]
            senY=[]
            pos=[]
            
            count+=1
            clear_output()
            print(count)
        if elem.tag=='wf':
            if elem.get('lemma').isalpha() or elem.get('lemma').isdigit():
                senX.append(elem.get('lemma'))
                senY.append(elem.get('lemma'))
                pos.append(elem.get('pos'))
        if elem.tag=='instance':
            
            
                senX.append(elem.get('lemma'))
            
            
                synset = wn.lemma_from_key(mapping[elem.get('id')][0] ).synset()
                synset_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos()
                senY.append(synset_id)
                pos.append(elem.get('pos'))
                
            
    with open(name+'.txt', 'wb') as f:
        for line in sentence:
            f.write((line+'\n').encode('utf-8'))
        
    with open('y'+name+'.txt', 'wb') as f:
        for line in Y:
            f.write((line+'\n').encode('utf-8')) 
          
    with open('pos'+name+'.txt', 'wb') as f:
        for line in POS:
            f.write((line+'\n').encode('utf-8'))


# In[3]:
#difference from Utils file function is that in this we pass actual file paths and also manage to build the vocabolary (in case we need to)
def processdata(filepath, goldpath, vocabname, name,bool):
    x=[]
    temp=[]
    with open(filepath, 'rb') as f:
        for line in f:
            temp=line.decode('utf-8').strip('\n').split()
            x.append(temp)
            
            
    y=[]
    temp=[]
    with open(goldpath, 'rb') as f:
        for line in f:
            temp=line.decode('utf-8').strip('\n').split()
            y.append(temp)
            
    vocabs=buildvocab(x,y,bool,vocabname)
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
    y_train=[]

    for sen in y:
        w=[]
        for elem in sen:
            if elem.isdigit():
                w.append(vocabs['<SUB>'])
            elif elem in  "!@#$%^&*()_-+={}[].,":
                w.append(vocabs['<SUB>'])
            elif elem not in vocabs:
                w.append(vocabs['<UNK>'])
            else:
                
                w.append(vocabs[elem])
        y_train.append(w)
    print('y finished')
    saveforgen(x_train,y_train,name)
    X_train=keras.preprocessing.sequence.pad_sequences(x_train,truncating='pre',padding='post',maxlen=30)
    Y_train=keras.preprocessing.sequence.pad_sequences(y_train,truncating='pre',padding='post',maxlen=30)
    Y_train2=Y_train.reshape(*Y_train.shape,1)
    return X_train, Y_train


# In[6]:
#function responsible for building the vocab. It iterates over x data, it samples it (min count set to 3) and 
#saves every instance from y data without fear of sampling out
def buildvocab(x,y,check, vocabname):
    
    if check==False:
        wordlist=[]
        occ=dict()
        for row in x:
            for elem in row:
                if elem.isalpha():
                    wordlist.append(elem)
                    if elem not in occ:
                        occ[elem]=0
                    else:
                        occ[elem]+=1
        print('part 1 vocab complete')            
        wordlist2=[]
        wordlist2.append('<PAD>')
        for word in wordlist:
            if occ[word]>3:
                if word not in wordlist2:
                    wordlist2.append(word)
        print('part 2 vocab complete')
        indexi=0
        for row in y:
            indexi+=1
            indexj=0
            for elem in row:
                indexj+=1
                if 'wn:'in elem:
                    if x[indexi-1][indexj-1] not in wordlist2:
                        wordlist2.append(x[indexi-1][indexj-1])
                    if elem not in wordlist2:
                        wordlist2.append(elem)
        print('part 3 vocab complete')
        wordlist2.append('<UNK>')
        wordlist2.append('<SUB>')
        
        vocabs=dict()
        for word in wordlist2:
            if word not in vocabs:
                vocabs[word]=len(vocabs)
                
        with open(vocabname+'.pickle','wb') as handle:
            pickle.dump(vocabs,handle)
            
        return vocabs
    else:
        with open(vocabname+'.pickle', 'rb') as handle:
            vocabs=pickle.load(handle)
        return vocabs


# In[7]:
#function needed to convert the data into data ready to use for the generator
def saveforgen(x,y,name):
    with open('x_'+name+'gen.txt', 'w') as f:
        for i in x:
            for elem in i:
                f.write((str(elem)+' '))
            f.write('\n')
    with open('y_'+name+'gen.txt', 'w') as f:
        for i in y:
            for elem in i:
                f.write((str(elem)+' '))
            f.write('\n')
  


# In[8]:
#the below mapping functions have same functionality as Utils
def buildmaps():
    babelnet=dict()
    prov=[]
    with open('../resources/babelnet2wordnet.tsv', 'r') as f:
        for line in f:
            prov=line.split()
            babelnet[prov[1]]=prov[0]
    domains=dict()
    prov=[]
    with open('../resources/babelnet2wndomains.tsv', 'r') as f:
        for line in f:
            prov=line.split()
            domains[prov[0]]=prov[1]
            
    lex=dict()
    prov=[]
    with open('../resources/babelnet2lexnames.tsv', 'r') as f:
        for line in f:
            prov=line.split()
            lex[prov[0]]=prov[1]
    return babelnet, domains, lex


# In[9]:

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


# In[10]:

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


# In[11]:

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
    #print('reached at finding max occ')
    #print(len(occ))
    if len(occ)!=0:    
        saveword=max(occ.items(), key=operator.itemgetter(1))[0]
    
        print('time to replace')
        for elem in index:
            yList[elem[0]][elem[1]]=saveword
    else:
        for elem in index:
            yList[elem[0]][elem[1]]='factotum'
    return yList  


# In[12]:

def get3outputs(datapath,name):
    y=[]
    with open(datapath, 'r') as handle:
        for line in handle:
            row=line.strip('\n').split()
            y.append(row)
    babelnet, domains, lex=buildmaps()
    y2=ybabel(y,babelnet)
    y3=ydomain(y2,domains)
    y4=ylex(y2,lex)
      

    with open('ybabel'+name+'.txt', 'wb') as f:
        for line in y2:
            f.write((' '.join(line)+'\n').encode('utf-8')) 
            
    with open('ydomain'+name+'.txt', 'wb') as d:
        for line in y3:
            d.write((' '.join(line)+'\n').encode('utf-8')) 
            
    with open('ylex'+name+'.txt', 'wb') as g:
        for line in y4:
            g.write((' '.join(line)+'\n').encode('utf-8')) 
    return y2,y3,y4


# In[ ]:
#this is a custom function used to save instances in the vocab by avoiding sampling , only usuable for domains and lexes
def buildvocabforlexdom(xpath,ypath,check, vocabname):
    
    x=[]
    temp=[]
    with open(xpath, 'rb') as f:
        for line in f:
            temp=line.decode('utf-8').strip('\n').split()
            x.append(temp)
            
            
    y=[]
    temp=[]
    with open(ypath, 'rb') as f:
        for line in f:
            temp=line.decode('utf-8').strip('\n').split()
            y.append(temp)
    
    if check==False:
        wordlist=[]
        occ=dict()
        for row in x:
            for elem in row:
                if elem.isalpha():
                    wordlist.append(elem)
                    if elem not in occ:
                        occ[elem]=0
                    else:
                        occ[elem]+=1
        print('part 1 vocab complete')            
        wordlist2=[]
        wordlist2.append('<PAD>')
        for word in wordlist:
            if occ[word]>3:
                if word not in wordlist2:
                    wordlist2.append(word)
        print('part 2 vocab complete')
        indexi=0
        for row in y:
            print(indexi)
            clear_output()
            indexi+=1
            indexj=0
            for elem in row:
                indexj+=1
                if elem not in wordlist:
                    if x[indexi-1][indexj-1] not in wordlist2:
                        wordlist2.append(x[indexi-1][indexj-1])
                    if elem not in wordlist2:
                        wordlist2.append(elem)
        print('part 3 vocab complete')
        wordlist2.append('<UNK>')
        wordlist2.append('<SUB>')
        
        vocabs=dict()
        for word in wordlist2:
            if word not in vocabs:
                vocabs[word]=len(vocabs)
                
        with open(vocabname+'.pickle','wb') as handle:
            pickle.dump(vocabs,handle)
            
        return vocabs
    else:
        with open(vocabname+'.pickle', 'rb') as handle:
            vocabs=pickle.load(handle)
        return vocabs


# In[ ]:




# In[12]:

#parsing and processing of train data


# In[ ]:

parsedata('../resources/semcor.data.xml','../resources/semcor.gold.key.txt', 'train')


# In[41]:

xtrain, ytrain=processdata('train.txt','ytrain.txt','finalvocab','train2', True)


# In[ ]:

#buildvocabforlexdom('train.txt', 'ydomaintrain.txt',False, 'domainvocab')
#buildvocabforlexdom('train.txt', 'ylextrain.txt',False, 'lexvocab')


# In[54]:

ytrainbabel,ytraindomain,ytrainlex=get3outputs('ytrain.txt','train2')


# In[57]:


junk, yd=processdata('train.txt', 'ydomaintrain2.txt', 'finalvocab', 'traindomain2' , True)
junk, yl=processdata('train.txt', 'ylextrain2.txt', 'finalvocab', 'trainlex2' , True)


# In[18]:

#validation data


# In[67]:

parsedata('../resources/semeval2015.data.xml','../resources/semeval2015.gold.key.txt', 'val')


# In[68]:

xval, yval=processdata('val.txt','yval.txt','vocab','val2', True)


# In[70]:

y2,y3,y4=get3outputs('yval.txt','val2')


# In[71]:


junk, yd=processdata('val.txt', 'ydomainval2.txt', 'domainvocab', 'valdomain2' , True)
junk, yl=processdata('val.txt', 'ylexval2.txt', 'lexvocab', 'vallex2' , True)


# In[ ]:

#test data


# In[54]:

parsedata('../resources/semeval2013.data.xml','../resources/semeval2013.gold.key.txt', 'val2013')


# In[55]:

xval, yval=processdata('val2013.txt','yval2013.txt','vocab','val2013', True)


# In[56]:

y2,y3,y4=get3outputs('yval2013.txt','val2013')


# In[15]:

junk, yb=processdata('val2013.txt', 'ybabelval2013.txt', 'babelvocab', 'val2013babel' , True)
junk, yd=processdata('val2013.txt', 'ydomainval2013.txt', 'finalvocab', 'val2013domain' , True)
junk, yl=processdata('val2013.txt', 'ylexval2013.txt', 'finalvocab', 'val2013lex' , True)


# In[57]:

junk, yb=processdata('val2013.txt', 'ybabelval2013.txt', 'babelvocab', 'val2013babel' , True)
junk, yd=processdata('val2013.txt', 'ydomainval2013.txt', 'domainvocab', 'val2013domain' , True)
junk, yl=processdata('val2013.txt', 'ylexval2013.txt', 'lexvocab', 'val2013lex' , True)


# In[58]:
#we can save the processed file into pickle files
import pickle
with open('xtest.pickle', 'wb') as handle:
    pickle.dump(xval, handle)
with open('ytest.pickle', 'wb') as handle:
    pickle.dump(yval, handle)


# In[45]:
#the below script is the main script used to calculate the length of each sentence of dataset, saving them into one file.
#it does it over all type of files (senses, lexes, domains) so the code is reused multiple times.
y=[]
temp=[]
with open('x_train2gen.txt', 'r') as f:
    for line in f:
        temp=line.strip('\n').split()
        y.append(temp)
lens=[]
lens.append(0)
count=0
i=0
for row in y:
    i+=1
    for w in row:
            
            count=len(w)+1+count
    lens.append(count+i)
lens.pop(-1)
print(lens[-1])
lenx=lens


# In[47]:

y=[]
temp=[]
with open('y_train2gen.txt', 'r') as f:
    for line in f:
        temp=line.strip('\n').split()
        y.append(temp)
lens=[]
lens.append(0)
count=0
i=0
for row in y:
    i+=1
    for w in row:
            
            count=len(w)+1+count
    lens.append(count+i)
lens.pop(-1)
print(lens[-1])
leny=lens


# In[60]:

y=[]
temp=[]
with open('y_traindomain2gen.txt', 'r') as f:
    for line in f:
        temp=line.strip('\n').split()
        y.append(temp)
lens=[]
lens.append(0)
count=0
i=0
for row in y:
    i+=1
    for w in row:
            
            count=len(w)+1+count
    lens.append(count+i)
lens.pop(-1)
print(lens[-1])
lend=lens


# In[62]:

y=[]
temp=[]
with open('y_trainlex2gen.txt', 'r') as f:
    for line in f:
        temp=line.strip('\n').split()
        y.append(temp)
lens=[]
lens.append(0)
count=0
i=0
for row in y:
    i+=1
    for w in row:
            
            count=len(w)+1+count
    lens.append(count+i)
lens.pop(-1)
print(lens[-1])
lenl=lens


# In[65]:

index=[]
index.append(lenx)
index.append(leny)
index.append(lend)
index.append(lenl)


# In[66]:

with open('indexes2.pickle', 'wb') as handle:
    pickle.dump(index, handle)


# In[73]:

y=[]
temp=[]
with open('x_val2gen.txt', 'r') as f:
    for line in f:
        temp=line.strip('\n').split()
        y.append(temp)
lens=[]
lens.append(0)
count=0
i=0
for row in y:
    i+=1
    for w in row:
            
            count=len(w)+1+count
    lens.append(count+i)
lens.pop(-1)
print(lens[-1])
lenx=lens
y=[]
temp=[]
with open('y_val2gen.txt', 'r') as f:
    for line in f:
        temp=line.strip('\n').split()
        y.append(temp)
lens=[]
lens.append(0)
count=0
i=0
for row in y:
    i+=1
    for w in row:
            
            count=len(w)+1+count
    lens.append(count+i)
lens.pop(-1)
print(lens[-1])
leny=lens
y=[]
temp=[]
with open('y_valdomain2gen.txt', 'r') as f:
    for line in f:
        temp=line.strip('\n').split()
        y.append(temp)
lens=[]
lens.append(0)
count=0
i=0
for row in y:
    i+=1
    for w in row:
            
            count=len(w)+1+count
    lens.append(count+i)
lens.pop(-1)
print(lens[-1])
lend=lens
y=[]
temp=[]
with open('y_vallex2gen.txt', 'r') as f:
    for line in f:
        temp=line.strip('\n').split()
        y.append(temp)
lens=[]
lens.append(0)
count=0
i=0
for row in y:
    i+=1
    for w in row:
            
            count=len(w)+1+count
    lens.append(count+i)
lens.pop(-1)
print(lens[-1])
lenl=lens


# In[74]:

index=[]
index.append(lenx)
index.append(leny)
index.append(lend)
index.append(lenl)


# In[75]:

with open('indexesval.pickle', 'wb') as handle:
    pickle.dump(index, handle)

