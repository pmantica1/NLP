
# coding: utf-8

# In[41]:


import pickle 
from metrics import AUCMeter
import torch
from tqdm import tqdm
import numpy as np


# In[88]:


indir = "similarity_scores/"
tfidf_validation_scores = pickle.load(open(indir+"validation_vectorizer_scores.pkl", "rb"))
tfidf_validation_similarities = pickle.load(open(indir+"validation_vectorizer_similarities.pkl", "rb"))
tfidf_testing_scores = pickle.load(open(indir+"testing_vectorizer_scores.pkl", "rb"))
tfidf_testing_similarities = pickle.load(open(indir+"testing_vectorizer_similarities.pkl", "rb"))


# In[53]:


cnn_validation_scores = pickle.load(open(indir+"validation_cnn_scores.pkl", "rb"))
cnn_validation_similarities = pickle.load(open(indir+"validation_cnn_similarities.pkl", "rb"))
cnn_testing_scores = pickle.load(open(indir+"testing_cnn_scores.pkl", "rb"))
cnn_testing_similarities = pickle.load(open(indir+"testing_cnn_similarities.pkl", "rb"))


# In[77]:

def process_tensor_list(tensor_list):
    return [ np.asscalar(tensor.numpy()[0]) for tensor in tensor_list]



cnn_validation_scores = process_tensor_list(cnn_validation_scores)
cnn_validation_similarities = process_tensor_list(cnn_validation_similarities)
cnn_testing_scores = process_tensor_list(cnn_testing_scores)
cnn_testing_similarities = process_tensor_list(cnn_testing_similarities)


# In[62]:





# In[14]:


len(tfidf_validation_scores)


# In[44]:


def compute_auc(scores_list, similarity_list):
    meter = AUCMeter()
    for score, similarity in tqdm(zip(scores_list, similarity_list)):
        meter.add(torch.FloatTensor([score]), torch.LongTensor([similarity]))
    print(meter.value(0.05))




# In[47]:


def normalize_scores(scalar_list):
    np_scalar_list = np.array(scalar_list)
    return list((np.array(scalar_list)-np.mean(np_scalar_list))/np.std(np_scalar_list))


# In[51]:


def avg(scalar_list1, scalar_list2, weight=0.5):
    return list(np.array(scalar_list1)*weight+np.array(scalar_list2)*(1-weight))


# In[71]:


compute_auc(tfidf_validation_scores, cnn_validation_similarities)


# In[48]:


compute_auc(normalize_scores(tfidf_testing_scores), tfidf_similarity_scores)


# In[45]:


compute_auc(process_tensor_list(cnn_validation_scores), process_tensor_list(cnn_validation_similarities))


# In[82]:


for i in range(11):
    weight = float(i)/10
    compute_auc(avg(normalize_scores(t), normalize_scores(cnn_validation_scores), weight=weight), cnn_validation_similarities)


# In[89]:


weight = 0.3
compute_auc(avg(normalize_scores(tfidf_validation_scores), normalize_scores(cnn_validation_scores), weight=weight), cnn_validation_similarities)

