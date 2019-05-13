#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import OrderedDict
import itertools as it
from functools import reduce 
import time


# # Helper Functions

# In[3]:


def k_fold_cross_validation(data, k):
    """
    params:
        data: This should be a pandas data frame
        k: This is an int indicating the folds of the data to be performed
           If k is 0, perform LOO cross-validation: TODO
           If k is 1, the data is both test and train
    output:
        train: This is a pandas data frame
        test: This is a pandas data frame
        cross: This is which kth fold that we have just yielded
    """
    if k == 1:
        yield(data, data, k)
        return()
    if k == 0: # TODO: double check
        for i, value in enumerate(data):
            train = data[:i].append(data[(i+1):])
            test = data[i:(i+1)]
            yield(train, test, i)
    size = len(data)
    for cross in range(k):
        start = int(cross*size/k)
        stop = int((cross+1)*size/k)
        train = data[:start].append(data[stop:])
        test = data[start:stop]
        yield(train,test,cross)


# In[4]:


def run_algorithm(train, test, algorithm):
    """
    params:
        train: a pandas data frame of the data
        test: a pandas data frame of the data
        algorithm: a pointer to a function that takes in data and outputs predictions
        #   params:
        #       data: a pandas dataframe
        #       info: information from training. If not present, train the model
        #   outputs:
        #       values: if training, it will output the parameters learned during training
        #               if testing, it will output the confusion_matrix
    outputs:
        confusion_matrix: The confusion matrix of the boolean classification
        duration: The amount of time that this took to run
    """
    start = time.time()
    training_info = algorithm(train)
    confusion_matrix = algorithm(test, training_info)
    duration = time.time() - start
    return(confusion_matrix, duration)


# In[5]:


def accuracy(confusion_matrix):
    """
    params:
        confusion_matrix: a dictionary where entries are of the form {(T/F,T/F):freq}
                          freq is the occurence of that prediction outcome
    ouputs:
        The output is a float between 0 and 1 indicating the overall accuracy of the \
        model given the binary confusion matrix.
    """
    correct = confusion_matrix[(True, True)]+confusion_matrix[(False, False)]
    return(correct/sum(list(confusion_matrix.values())))


# In[6]:


def add_term(dicto, word, sarcasm):
        if not word in dicto:
            dicto[word] = [0,0]
            dicto[word][sarcasm] = 1  #if this is the first time we see the word, make its count the step
        else:
            dicto[word][sarcasm] += 1 #increment the count with another sighting


# # Model Functions

# In[7]:


def prior(data,info=None):
    values = None
    if info is None:
        prior=sum(data['is_sarcastic'])/len(data)
        values = prior
    else:
        answer = 0
        # If 1 is more common than 0, guess 1.
        if info > 0.5:
            answer = 1
        confusion_matrix = {x:0 for x in it.product([0,1],repeat=2)}
        for x in data['is_sarcastic']:
            confusion_matrix[(answer,x)] += 1
        values = confusion_matrix
    return(values)


# In[8]:


def NB(data,info=None):
    values = None
    counts = {}
    if info is None:
        for entry, sarc in zip(data['headline'],data['is_sarcastic']):
            for word in entry.split(" "):
                add_term(counts, word, sarc)
        v=list(it.chain(*list(counts.values())))
        num_serious_words=sum(v[0::2])
        num_sarcastic_words=sum(v[1::2])
        for word in counts:
            counts[word][0] /= num_serious_words
            counts[word][1] /= num_sarcastic_words
        values = counts
    else:
        confusion_matrix = {x:0 for x in it.product([0,1],repeat=2)}
        for entry, sarc in zip(data['headline'],data['is_sarcastic']):
            # For every headline, multiply the frequency of each word for each class
            # If a word in the test set is not found in the training set, ignore it
            r = list(it.chain(*[info.get(word, [1,1]) for word in entry.split(" ")]))
            p_serious = reduce((lambda x, y: x * y), r[0::2]) 
            p_sarcasm = reduce((lambda x, y: x * y), r[1::2]) 
            result=(p_serious < p_sarcasm, bool(sarc))
            confusion_matrix[result] += 1
        values = confusion_matrix
    return(values)


# In[9]:


def NB_rank(data,info=None):
    values = None
    counts = {}
    if info is None:
        for entry, sarc in zip(data['headline'],data['is_sarcastic']):
            for word in entry.split(" "):
                add_term(counts, word, sarc)
        # Get the ranks of each word in reference to serious
        ordered_counts_ser=OrderedDict(sorted(counts.items(), reverse=True, key=lambda x: x[1][0]))
        for i,count in enumerate(ordered_counts_ser):
            ordered_counts_ser[count][0]=i
        # Get the ranks of each word in reference to sarcasm
        ordered_counts_sarc=OrderedDict(sorted(ordered_counts_ser.items(), reverse=True, key=lambda x: x[1][1]))
        for i,count in enumerate(ordered_counts_sarc):
            ordered_counts_sarc[count][1]=i
        values = ordered_counts_sarc
    else:
        confusion_matrix = {x:0 for x in it.product([0,1],repeat=2)}
        for entry, sarc in zip(data['headline'],data['is_sarcastic']):
            # If a word in the test set is not found in the training set, ignore it
            r=list(it.chain(*[info.get(word,[0,0]) for word in entry.split(" ")]))
            # For every headline, add up the rankings of each word for each class
            psar = sum(r[1::2])
            pser = sum(r[0::2])
            result=(pser > psar,bool(sarc))
            confusion_matrix[result] += 1
        values = confusion_matrix
    return(values)


# # Let's Test the Classifiers!
# 
# ### Let's Get the Data First

# In[10]:

if __name__ == "__main__":
    data=pd.read_json('../data/Sarcasm_Headlines_Dataset.json', lines=True)
    data[['headline','is_sarcastic']].head()


    # The next couple parts are borrowed from work by Tanumoy Nandy at https://www.kaggle.com/tanumoynandy/sarcasm-detection-lstm.

    # In[11]:


    # Get prior proportions
    sns.countplot(data.is_sarcastic)
    plt.xlabel('Label')
    plt.title('Sarcasm vs Non-sarcasm')


    # In[12]:


    # remove upper case, weird white space and punctuation
    data['headline'] = data['headline'].apply(lambda x: x.lower())
    data['headline'] = data['headline'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    data[['headline','is_sarcastic']].head()


    # ### Let's Now Run the Algorithms

    # In[13]:


    # prior
    acc = 0
    k=10
    for train, test, cross in k_fold_cross_validation(data, k): # never before seen words
        confusion_matrix, duration = run_algorithm(train, test, prior)
        acc += accuracy(confusion_matrix)
    print(acc/k)


    # In[14]:


    # NB
    acc = 0
    k=10
    for train, test, cross in k_fold_cross_validation(data, k): # never before seen words
        confusion_matrix, duration = run_algorithm(train, test, NB)
        acc += accuracy(confusion_matrix)
    print(acc/k)


    # In[15]:


    # NB_rank
    acc = 0
    k=5
    for train, test, cross in k_fold_cross_validation(data, k): # never before seen words
        confusion_matrix, duration = run_algorithm(train, test, NB_rank)
        acc += accuracy(confusion_matrix)
    print(acc/k)


    # It seems that worst case scenario, we will get 0.56 accuracy through guessing one value all the time. However, if we use NB, we can achieve an accuracy of around 0.77. This is a C+.
    # 
    # If we want to generate headlines, we will want a classifier that does even better. I will arbitrarily set the threshold to 95%. Why? Because when we give NB the opportunity to memorize, it can achieve 96% accuracy.
    # 
    # Notice that we are only removing non-alphanumerics and uppercase, but we are not removing things like conjugation for tense or number. We are also not removing stop words.

    # # NLTK

    # In[48]:


    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')


    # In[49]:


    from nltk.corpus import stopwords
    stop=list(stopwords.words('english'))[:]


    # In[50]:


    sentence="First shalt thou take out the Holy Pin. Then, shalt thou count to three, no more, no less.".lower()
    [w for w in word_tokenize(sentence) if not w in stop]


    # ### Classifiers with nltk

    # In[55]:


    def NB_nltk(data,info=None):
        values = None
        counts = {}
        if info is None:
            for entry, sarc in zip(data['headline'],data['is_sarcastic']):
                for word in entry.split(" "):
                    if not word in stop:
                        add_term(counts, word, sarc)
            v=list(it.chain(*list(counts.values())))
            num_serious_words=sum(v[0::2])
            num_sarcastic_words=sum(v[1::2])
            for word in counts:
                counts[word][0] /= num_serious_words
                counts[word][1] /= num_sarcastic_words
            values = counts
        else:
            confusion_matrix = {x:0 for x in it.product([0,1],repeat=2)}
            for entry, sarc in zip(data['headline'],data['is_sarcastic']):
                # For every headline, multiply the frequency of each word for each class
                # If a word in the test set is not found in the training set, ignore it
                r = list(it.chain(*[info.get(word, [1,1]) for word in entry.split(" ")]))
                p_serious = reduce((lambda x, y: x * y), r[0::2]) 
                p_sarcasm = reduce((lambda x, y: x * y), r[1::2]) 
                result=(p_serious < p_sarcasm, bool(sarc))
                confusion_matrix[result] += 1
            values = confusion_matrix
        return(values)


    # In[56]:


    def NB_rank_nltk(data,info=None):
        values = None
        counts = {}
        if info is None:
            for entry, sarc in zip(data['headline'],data['is_sarcastic']):
                for word in entry.split(" "):
                    if not word in stop:
                        add_term(counts, word, sarc)
            # Get the ranks of each word in reference to serious
            ordered_counts_ser=OrderedDict(sorted(counts.items(), reverse=True, key=lambda x: x[1][0]))
            for i,count in enumerate(ordered_counts_ser):
                ordered_counts_ser[count][0]=i
            # Get the ranks of each word in reference to sarcasm
            ordered_counts_sarc=OrderedDict(sorted(ordered_counts_ser.items(), reverse=True, key=lambda x: x[1][1]))
            for i,count in enumerate(ordered_counts_sarc):
                ordered_counts_sarc[count][1]=i
            values = ordered_counts_sarc
        else:
            confusion_matrix = {x:0 for x in it.product([0,1],repeat=2)}
            for entry, sarc in zip(data['headline'],data['is_sarcastic']):
                # If a word in the test set is not found in the training set, ignore it
                r=list(it.chain(*[info.get(word,[0,0]) for word in entry.split(" ")]))
                # For every headline, add up the rankings of each word for each class
                psar = sum(r[1::2])
                pser = sum(r[0::2])
                result=(pser > psar,bool(sarc))
                confusion_matrix[result] += 1
            values = confusion_matrix
        return(values)


    # In[57]:


    # NB
    acc = 0
    k=10
    for train, test, cross in k_fold_cross_validation(data, k): # never before seen words
        confusion_matrix, duration = run_algorithm(train, test, NB_nltk)
        acc += accuracy(confusion_matrix)
    print(acc/k)


    # In[58]:


    # NB_rank
    acc = 0
    k=10
    for train, test, cross in k_fold_cross_validation(data, k): # never before seen words
        confusion_matrix, duration = run_algorithm(train, test, NB_rank_nltk)
        acc += accuracy(confusion_matrix)
    print(acc/k)


    # It appears that removing the stopwords reduces accuracy, but not by much. This is an interesting finding. Let's see if we can get better accuracy using a neural network!




