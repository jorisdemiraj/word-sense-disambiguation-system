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
import Model
import Utils
from Model import buildmodel
from Utils import ybabel , ydomain, ylex, processing,modindexing

def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    #the Utils file functions are used to parse and process the data
    xdata,babelnet,domains,lex,vocab,sentences, pos,ids, indexes=processing(input_path,resources_path)
    indexes=modindexing(indexes)
    indexes=modindexing(indexes)
    indexes=modindexing(indexes)
    invocab={v:k for k,v in vocab.items()}
    model=buildmodel()
    #load the weights
    model.load_weights(resources_path+'modelweightstestsplit.h5')

    #quick function to get wordnet id from synset
    def wn_id_from_synset(synset):

        offset = str(synset.offset())
        offset = "0" * (8 - len(offset)) + offset  
        wn_id = "wn:%s%s" % (offset, synset.pos())

        return wn_id
    #candidate function is used to extract the instance candidates which will later be used to limit the argmax over these candidates.
    #it uses the lemma and POS extracted from parsing to return a list of senses, we keep only whats in the vocabolary.
    def  candidates(xdata,pos, ids):


        pos_vocab = {"ADJ": wn.ADJ, "ADV": wn.ADV, "NOUN": wn.NOUN, "VERB": wn.VERB} 
        candidate=dict()
        synset=[]
        indexi=0
        for row in indexes:

            indexi+=1
            if pos[row[0]][row[1]] in pos_vocab:
                synsets=wn.synsets(xdata[row[0]][row[1]],pos=[pos_vocab[pos[row[0]][row[1]]]])
            else:
                synsets=wn.synsets(xdata[row[0]][row[1]])

            if len(synsets) != 0:
                candidate[indexi-1]=[wn_id_from_synset(syn) for syn in synsets]
            else:
                candidate[indexi-1]=xdata[row[0]][row[1]]


        return candidate



    candidate=candidates(sentences,pos, ids)

    print('candidates loaded')


    # In[ ]:
    #this function has one role. It converts the candidates to babelnet, lexes or domains . It is used in all 3 prediction types.
    #It also converts to numerical form
    listc=[]
    listwn=[]

    for row in range(len(indexes)):
        prov=[]
        prov2=[]
        for t in candidate[row]:

            if t in vocab:
                prov.append(t)
                prov2.append(vocab[t])

        listwn.append(prov)
        listc.append(prov2)
    listb=ybabel(listwn,babelnet)      
    listd=ydomain(listb,domains)  
    listl=ylex(listb,lex)  
    print('time to predict')   


    # In[ ]:
    #numerical form conversion for domain candidates
    outer=[]
    for elem in listd:
        inner=[]
        if elem!=[]:
            for word in elem:
                if word in vocab:
                    inner.append(vocab[word])

        outer.append(inner)
    listdm=outer


    # In[ ]:
    #numerical form conversion for lex candidates
    outer=[]
    for elem in listl:
        inner=[]
        if elem!=[]:
            for word in elem:
                if word in vocab:
                    inner.append(vocab[word])

        outer.append(inner)
    listlx=outer


    # In[ ]:
    #predicting
    y_predict, y_predict2, y_predict3 = model.predict(xdata)
    print('passed prediction phase')



   
    y_predicts = np.argmax(y_predict, axis=2)
    y_predicts2 = np.argmax(y_predict2, axis=2)
    y_predicts3 = np.argmax(y_predict3, axis=2)

    print('stuff predicted')



    # In[ ]:


    #we argmax over the candidates present in the probability distribution
    #we use the position of each instance to keep track of which word to replace in case of non sense prediction
    
    nr=0
    replace=[]
    mfs=[]
    for elem in indexes:
        nr+=1
        if elem[1]>=30 :
            mfs.append(elem)
        elif y_predict[elem[0]][elem[1]][listc[nr-1]].size==0:
            replace.append(elem)
            y_predicts[elem[0]][elem[1]]= len(invocab)+1
        else:
            y_predicts[elem[0]][elem[1]]=listc[nr-1][np.argmax(y_predict[elem[0]][elem[1]][listc[nr-1]])]

    print('done')

    ypred=[]
    count=0
    #check which instance requires to execute the MFS solution. We use indexes saved from the above function
    for row in y_predicts:
        line=[]
        for word in row:
            if word==len(invocab)+1:

                synsets = wn.synsets(sentences[replace[count][0]][replace[count][1]])


                if synsets is None or len(synsets) == 0:
                    line.append(sentences[replace[count][0]][replace[count][1]])
                  
                synset = synsets[0]                         # fetch MFS
                line.append(wn_id_from_synset(synset))
                count+=1
            else:
                 line.append(invocab[word])
        ypred.append(line)


    #convert to babelnet
    ypred=ybabel(ypred,babelnet)
    #save to file and also a backup check on MFS
    with open(output_path+'predictionsense.txt', 'w')as f:
        for i in range(len(indexes)):
            if indexes[i] in mfs:


                synsets = wn.synsets(sentences[indexes[i][0]][indexes[i][1]])


                if synsets is None or len(synsets) == 0:
                    f.write(ids[i][0]+' '+sentences[indexes[i][0]][indexes[i][1]]+'\n')
                   
                else:
                    synset = synsets[0]                         # fetch MFS
                    f.write(ids[i][0]+' '+wn_id_from_synset(synset)+'\n')
            else:
                f.write(ids[i]+' '+ypred[indexes[i][0]][indexes[i][1]]+'\n')



#all the functions and codes of the domains and lex predictions use the same concept and code used for senses.
#The code used for the sense prediction is a generalized code, appropriate to use to return senses, domains or lexes.
#Only difference to notice is the ouput file name and the lack of use of MFS
def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    xdata,babelnet,domains,lex,vocab,sentences, pos,ids, indexes=processing(input_path,resources_path)
    indexes=modindexing(indexes)
    indexes=modindexing(indexes)
    indexes=modindexing(indexes)
    invocab={v:k for k,v in vocab.items()}


    # In[ ]:

    model=buildmodel()


    # In[ ]:

    #from keras.models import load_model,load_weights
    model.load_weights(resources_path+'modelweightstestsplit.h5')
    #model.load_weights('finalweights.h5')

    # In[ ]:

    def wn_id_from_synset(synset):

        offset = str(synset.offset())
        offset = "0" * (8 - len(offset)) + offset  # append heading 0s to the offset
        wn_id = "wn:%s%s" % (offset, synset.pos())

        return wn_id

    def  candidates(xdata,pos, ids):


        pos_dictionary = {"ADJ": wn.ADJ, "ADV": wn.ADV, "NOUN": wn.NOUN, "VERB": wn.VERB} 
        candidate=dict()
        synset=[]
        indexi=0
        for row in indexes:

            indexi+=1
            if pos[row[0]][row[1]] in pos_dictionary:
                synsets=wn.synsets(xdata[row[0]][row[1]],pos=[pos_dictionary[pos[row[0]][row[1]]]])
            else:
                synsets=wn.synsets(xdata[row[0]][row[1]])

            if len(synsets) != 0:
                candidate[indexi-1]=[wn_id_from_synset(syn) for syn in synsets]
            else:
                candidate[indexi-1]=xdata[row[0]][row[1]]


        return candidate



    candidate=candidates(sentences,pos, ids)

    print('candidates loaded')


    # In[ ]:

  
    listwn=[]

    for row in range(len(indexes)):
        prov=[]
 
        for t in candidate[row]:

            if t in vocab:
                prov.append(t)
                
        listwn.append(prov)
     
    listb=ybabel(listwn,babelnet)      
    listd=ydomain(listb,domains)   
    print('time to predict')   


    # In[ ]:

    outer=[]
    for elem in listd:
        inner=[]
        if elem!=[]:
            for word in elem:
                if word in vocab:
                    inner.append(vocab[word])

        outer.append(inner)
    listdm=outer



    # In[ ]:
    y_predict, y_predict2, y_predict3 = model.predict(xdata)
    print('passed prediction phase')




    y_predicts2 = np.argmax(y_predict2, axis=2)
    print('stuff predicted')



    # In[ ]:


    ################DOMAINS###############################
    nr=0
    replace=[]
    mfs=[]
    for elem in indexes:
        nr+=1
        if elem[1]>=30 :
            mfs.append(elem)
        elif y_predict2[elem[0]][elem[1]][listdm[nr-1]].size==0:

            y_predicts2[elem[0]][elem[1]]= vocab['<UNK>']
        else:
            y_predicts2[elem[0]][elem[1]]=listdm[nr-1][np.argmax(y_predict2[elem[0]][elem[1]][listdm[nr-1]])]

    
    
    #get occ:
    occ=dict()
    for elem in indexes:
        v=invocab[y_predicts2[elem[0]][elem[1]]]
        if v!='<UNK>':
            if v not in occ:
                occ[v]=1
            else:
                occ[v]=occ[v]+1

    mostseen=max(occ.items(), key=operator.itemgetter(1))[0]
    print(mostseen)

    ypred2=[]
    for row in y_predicts2:
        line=[]
        for word in row:
            if invocab[word]!='<UNK>':
                line.append(invocab[word])
            else:
                line.append(mostseen)
        ypred2.append(line)  

    with open(output_path+'predictionsdomain.txt', 'w')as f:
        for i in range(len(indexes)):

            f.write(ids[i]+' '+ypred2[indexes[i][0]][indexes[i][1]]+'\n')

    ###################################################
    



def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    
    xdata,babelnet,domains,lex,vocab,sentences, pos,ids, indexes=processing(input_path,resources_path)
    indexes=modindexing(indexes)
    indexes=modindexing(indexes)
    indexes=modindexing(indexes)
    invocab={v:k for k,v in vocab.items()}


    # In[ ]:

    model=buildmodel()


    # In[ ]:

    #from keras.models import load_model,load_weights
    model.load_weights(resources_path+'modelweightstestsplit.h5')
    #model.load_weights('finalweights.h5')

    # In[ ]:

    def wn_id_from_synset(synset):

        offset = str(synset.offset())
        offset = "0" * (8 - len(offset)) + offset  # append heading 0s to the offset
        wn_id = "wn:%s%s" % (offset, synset.pos())

        return wn_id

    def  candidates(xdata,pos, ids):


        pos_dictionary = {"ADJ": wn.ADJ, "ADV": wn.ADV, "NOUN": wn.NOUN, "VERB": wn.VERB} 
        candidate=dict()
        synset=[]
        indexi=0
        for row in indexes:

            indexi+=1
            if pos[row[0]][row[1]] in pos_dictionary:
                synsets=wn.synsets(xdata[row[0]][row[1]],pos=[pos_dictionary[pos[row[0]][row[1]]]])
            else:
                synsets=wn.synsets(xdata[row[0]][row[1]])

            if len(synsets) != 0:
                candidate[indexi-1]=[wn_id_from_synset(syn) for syn in synsets]
            else:
                candidate[indexi-1]=xdata[row[0]][row[1]]


        return candidate



    candidate=candidates(sentences,pos, ids)

    print('candidates loaded')


    # In[ ]:

   
    listwn=[]

    for row in range(len(indexes)):
        prov=[]
       
        for t in candidate[row]:

            if t in vocab:
                prov.append(t)
               


        listwn.append(prov)
 
    listb=ybabel(listwn,babelnet)      
    
    listl=ylex(listb,lex)  
    print('time to predict')   




    # In[ ]:

    outer=[]
    for elem in listl:
        inner=[]
        if elem!=[]:
            for word in elem:
                if word in vocab:
                    inner.append(vocab[word])

        outer.append(inner)
    listlx=outer


    # In[ ]:
    y_predict, y_predict2, y_predict3 = model.predict(xdata)
    print('passed prediction phase')




    y_predicts3 = np.argmax(y_predict3, axis=2)

    print('stuff predicted')



    # In[ ]:


    ################LEXES###############################


    nr=0
    replace=[]
    mfs=[]
    for elem in indexes:
        nr+=1
        if elem[1]>=30 :
            mfs.append(elem)
        elif y_predict3[elem[0]][elem[1]][listlx[nr-1]].size==0:
            mfs.append(elem)

        else:
            y_predicts3[elem[0]][elem[1]]=listlx[nr-1][np.argmax(y_predict3[elem[0]][elem[1]][listlx[nr-1]])]

    print('done')


    #this part of the code is responsible for finding the most common lex predicted and replacing the UNK predictions
    #in return we get a lex only prediction file
    occ=dict()
    for elem in indexes:
        v=invocab[y_predicts3[elem[0]][elem[1]]]
        if v!='<UNK>':
            if v not in occ:
                occ[v]=1
            else:
                occ[v]=occ[v]+1

    mostseen=max(occ.items(), key=operator.itemgetter(1))[0]
    print(mostseen)



    ypred3=[]
    for row in y_predicts3:
        line=[]
        for word in row:
                if invocab[word]!='<UNK>':
                    line.append(invocab[word])
                else:
                    line.append(mostseen)
                    
        ypred3.append(line)
   
    with open(output_path+'predictionslex.txt', 'w')as f:
        for i in range(len(indexes)):

            f.write(ids[i]+' '+ypred3[indexes[i][0]][indexes[i][1]]+'\n')


