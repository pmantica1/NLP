import pickle 
from nn_utils import test_auc_step 
from database import AndroidDatabase 
from torch.utils import data 
from load_model import load_cnn_encoder
from tqdm import tqdm

def compute_scores_and_similarities(model, dataset):
    data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    score_list = [] 
    similarity_list = [] 
    for batch in tqdm(data_loader):
        score, similarity = test_auc_step(model, batch)
        score_list.append(score)
        similarity_list.append(similarity)
    return score_list, similarity_list

if __name__ == "__main__":
    android_database = AndroidDatabase(use_glove=True)
    validation_set = android_database.get_validation_dataset()
    testing_set = android_database.get_testing_dataset()

    filepath = "cnn_encoder.pt"
    feature_vector_dimensions = 300
    questions_vector_dimensions = 500
    kernel_size = 3
    encoder = load_cnn_encoder(filepath, feature_vector_dimensions, questions_vector_dimensions, kernel_size)

    validation_scores, validation_similarities = compute_scores_and_similarities(encoder, validation_set)
    testing_scores, testing_similarities = compute_scores_and_similarities(encoder, testing_set)
    pickle.dump(validation_scores, open("validation_vectorizer_scores.pkl", "wb"))
    pickle.dump(validation_similarities, open("validation_vectorizer_similarities.pkl", "wb"))
    pickle.dump(testing_scores, open("testing_vectorizer_scores.pkl", "wb"))
    pickle.dump(testing_similarities, open("testing_vectorizer_similarities.pkl", "wb"))
    
