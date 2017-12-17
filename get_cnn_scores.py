import pickle 
from nn_utils import test_auc_step 
from database import AndroidDatabase 
from torch import data 

def compute_scores_and_similarities(model, dataset):
    data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    score_list = [] 
    similarity_list = [] 
    for batch in tqdm(data_loader):
        score, similarity = test_auc_step(nn_model, batch)
        score_list.append(score)
        similarity_list.append(similarity)
    return score_list, similarity_list

if __name__ == "__main__":
	android_database = database.AndroidDatabase(use_glove=True)
    validation_set = android_database.get_validation_dataset()
    testing_set = android_database.get_testing_dataset()
    #LOAD MODEL HERE 
    validation_scores, validation_similarities = compute_scores_and_similarities(# ADD MODEL HERE, validation_set)
    testing_scores, testing_similarities = compute_scores_and_similarities(# ADD MODEL HERE, testing_set)
    pickle.dump(validation_scores, open("validation_vectorizer_scores.pkl", "wb"))
    pickle.dump(validation_similarities, open("validation_vectorizer_similarities.pkl", "wb"))
    pickle.dump(testing_scores, open("testing_vectorizer_scores.pkl", "wb"))
    pickle.dump(testing_similarities, open("testing_vectorizer_similarities.pkl", "wb"))
    
