
# coding: utf-8

# In[22]:


import pickle 
from metrics import AUCMeter
import torch
from tqdm import tqdm
import numpy as np
import json


# In[23]:


indir = "similarity_scores/"
tfidf_validation_scores = json.load(open(indir+"validation_vectorizer_scores.json", "r"))
tfidf_validation_similarities = json.load(open(indir+"validation_vectorizer_similarities.json", "r"))
tfidf_testing_scores = json.load(open(indir+"testing_vectorizer_scores.json", "r"))
tfidf_testing_similarities = json.load(open(indir+"testing_vectorizer_similarities.json", "rb"))


# In[30]:


def process_tensor_list(tensor_list):
    return [ np.asscalar(tensor.numpy()[0]) for tensor in tensor_list]


# In[3]:


cnn_validation_scores = pickle.load(open(indir+"validation_cnn_scores.pkl", "rb"))
cnn_validation_similarities = pickle.load(open(indir+"validation_cnn_similarities.pkl", "rb"))
cnn_testing_scores = pickle.load(open(indir+"testing_cnn_scores.pkl", "rb"))
cnn_testing_similarities = pickle.load(open(indir+"testing_cnn_similarities.pkl", "rb"))


# In[6]:


cnn_validation_scores = process_tensor_list(cnn_validation_scores)
cnn_validation_similarities = process_tensor_list(cnn_validation_similarities)
cnn_testing_scores = process_tensor_list(cnn_testing_scores)
cnn_testing_similarities = process_tensor_list(cnn_testing_similarities)







def compute_auc(scores_list, similarity_list):
    meter = AUCMeter()
    for score, similarity in tqdm(zip(scores_list, similarity_list)):
        meter.add(torch.FloatTensor([score]), torch.LongTensor([similarity]))
    print(meter.value(0.05))


# In[5]:





# In[8]:


def normalize_scores(scalar_list):
    np_scalar_list = np.array(scalar_list)
    return list((np.array(scalar_list)-np.mean(np_scalar_list))/np.std(np_scalar_list))


# In[9]:


def avg(scalar_list1, scalar_list2, weight=0.5):
    return list(np.array(scalar_list1)*weight+np.array(scalar_list2)*(1-weight))


# In[27]:


def opt_max(scalar_list1, scalar_list2, weight=0.5):
    return list(np.maximum(np.array(scalar_list1), np.array(scalar_list2)))



weight = 0.3
compute_auc(avg(normalize_scores(tfidf_validation_scores), normalize_scores(cnn_validation_scores), weight=weight), cnn_validation_similarities)
