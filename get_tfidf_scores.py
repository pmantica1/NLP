from database import AndroidDatabase
import metrics
from sklearn.metrics.pairwise import paired_cosine_distances as cosine
from tqdm import tqdm
import torch
from torch.utils import data
import json 

def compute_scores_and_similarities(dataset):
    score_list = [] 
    similarity_list = [] 
    for i in tqdm(range(len(dataset))):
        query_pair = dataset[i]
        question1_vec = (query_pair["id_1_title_vec"] + query_pair["id_1_body_vec"]) / 2.0
        question2_vec = (query_pair["id_2_title_vec"] + query_pair["id_2_body_vec"]) / 2.0
        score = (1 - cosine(question1_vec.numpy(), question2_vec.numpy()))[0]
        similarity = query_pair["similarity"]
        score_list.append(score)
        similarity_list.append(similarity)
    return score_list, similarity_list

if __name__ == "__main__":
    android_database = AndroidDatabase(use_count_vectorizer=True)
    validation_set = android_database.get_validation_dataset()
    testing_set = android_database.get_testing_dataset()
    validation_scores, validation_similarities = compute_scores_and_similarities(validation_set)
    testing_scores, testing_similarities = compute_scores_and_similarities(testing_set)
    json.dump(validation_scores, open("validation_vectorizer_scores.json", "w"))
    json.dump(validation_similarities, open("validation_vectorizer_similarities.json", "w"))
    json.dump(testing_scores, open("testing_vectorizer_scores.json", "w"))
    json.dump(testing_similarities, open("testing_vectorizer_similarities.json", "w"))
    
