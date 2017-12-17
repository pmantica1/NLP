from torch.utils import data
import torch
from tqdm import tqdm
from nn_utils import AdversarialLoss, evaluate_multi_questions
from cnn import CNN
from ffnn import FFNN
from lstm import LSTM
from nn_utils import test_auc, GRL
from database import TransferLearningDatabase, UbuntuDatabase
import random



TITLE_VEC = "title_vec"
BODY_VEC = "body_vec"
SIM_TITLE_VEC = "sim_title_vec"
SIM_BODY_VEC = 'sim_body_vec'
RAND_TITLE_VECS = "rand_title_vecs"
RAND_BODY_VECS = "rand_body_vecs"
UBUNTU_RAND_TITLE_VECS = "ubuntu_rand_title_vecs"
UBUNTU_RAND_BODY_VECS = "ubuntu_rand_body_vecs"
ANDROID_RAND_TITLE_VECS = "android_rand_title_vecs"
ANDROID_RAND_BODY_VECS = "android_rand_body_vecs"


def train_epoch(encoder, classifier, grl, dataset, optimizer_encoder, optimizer_domain, batch_size, lamb):
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for batch in tqdm(data_loader):
        train_step(encoder, classifier, grl, batch, optimizer_encoder, optimizer_domain, lamb)


def train_step(encoder, classifier, grl, batch, optimizer_encoder, optimizer_domain, lamb):
    loss_fn = AdversarialLoss(lamb)

    #get params from batch
    questions_title_batch = batch[TITLE_VEC]
    questions_body_batch = batch[BODY_VEC]

    similar_questions_title_batch = batch[SIM_TITLE_VEC]
    similar_questions_body_batch = batch[SIM_BODY_VEC]

    negative_questions_title_batch = batch[RAND_TITLE_VECS]
    negative_questions_body_batch = batch[RAND_BODY_VECS]

    ubuntu_rand_questions_title_batch = batch[UBUNTU_RAND_TITLE_VECS]
    ubuntu_rand_questions_body_batch = batch[UBUNTU_RAND_BODY_VECS]

    android_rand_questions_title_batch = batch[ANDROID_RAND_TITLE_VECS]
    android_rand_questions_body_batch = batch[ANDROID_RAND_BODY_VECS]


    #evaluate encoder
    questions_batch = encoder.evaluate(questions_title_batch, questions_body_batch)
    similar_questions_batch = encoder.evaluate(similar_questions_title_batch, similar_questions_body_batch)
    negative_questions_batch = evaluate_multi_questions(encoder, negative_questions_title_batch, negative_questions_body_batch)

    ubuntu_questions_batch = evaluate_multi_questions(encoder, ubuntu_rand_questions_title_batch, ubuntu_rand_questions_body_batch)
    android_questions_batch = evaluate_multi_questions(encoder, android_rand_questions_title_batch, android_rand_questions_body_batch)

    #evaluate classifier
    ubuntu_labels_probabilities = torch.cat([classifier(ubuntu_questions_batch[:,:,i]).unsqueeze(2) for i in xrange(ubuntu_questions_batch.data.shape[2])], dim=2)
    android_labels_probabilities = torch.cat([classifier(android_questions_batch[:,:,i]).unsqueeze(2) for i in xrange(android_questions_batch.data.shape[2])], dim=2)

    #print ubuntu_labels_probabilities

    #get loss
    loss_encoder, loss_domain = loss_fn(questions_batch, similar_questions_batch, negative_questions_batch, ubuntu_labels_probabilities, android_labels_probabilities)

    #print loss_encoder
    #print loss_domain

    optimizer_encoder.zero_grad()
    loss_encoder.backward(retain_graph=True)
    optimizer_encoder.step()

    optimizer_domain.zero_grad()
    loss_domain.backward()
    optimizer_domain.step()



if __name__ == "__main__":
    feature_vector_dimensions = 300
    questions_vector_dimensions = 500
    kernel_size = 3

    classifier_hidden_size_1 = 300
    classifier_hidden_size_2 = 150
    num_labels = 2

    learning_rate = 1e-4
    weight_decay = 1e-3
    n_epochs = 4
    batch_size = 16

    encoder = CNN(feature_vector_dimensions, questions_vector_dimensions, kernel_size).cuda()
    classifier = FFNN(questions_vector_dimensions, classifier_hidden_size_1, classifier_hidden_size_2, num_labels).cuda()

    lamb_list = [1e-1] 
    best_lamb = 0 
    best_score = 0 

    database = TransferLearningDatabase()
    #ubuntu_database = UbuntuDatabase(use_glove=True)
    for lamb in lamb_list: 
        training_dataset = database.get_training_set()
        android_validation_dataset = database.get_validation_set()
        android_test_dataset = database.get_testing_set()

        #ubuntu_validation_dataset = ubuntu_database.get_validation_dataset()
        #ubuntu_testing_dataset = ubuntu_database.get_testing_dataset()

        grl = GRL(lamb)

        optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer_domain = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in xrange(n_epochs):
            train_epoch(encoder, classifier, grl, training_dataset, optimizer_encoder, optimizer_domain, batch_size, lamb)
            score = test_auc(encoder, android_validation_dataset)
            print score
            #print test(encoder, ubuntu_validation_dataset)
        
        if score > best_score:
            best_score = score
            best_lamb = lamb 
        print(best_score)
        print(best_lamb)

    print test_auc(encoder, android_test_dataset)

    filepath = 'cnn_encoder.pt'
    torch.save(encoder.cpu().state_dict(), filepath)