from torch.utils import data
import torch
from tqdm import tqdm
from nn_utils import AdversarialLoss, evaluate_multi_questions
from cnn import CNN
from ffnn import FFNN
from lstm import LSTM
from nn_utils import test_auc
from database import TransferLearningDatabase



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


def train_epoch(encoder, classifier, dataset, optimizer_encoder, optimizer_domain, batch_size, lamb):
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for batch in tqdm(data_loader):
        train_step(encoder, classifier, batch, optimizer_encoder, optimizer_domain, lamb)


def train_step(encoder, classifier, batch, optimizer_encoder, optimizer_domain, lamb):
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

    #initialize gradients of optimizers
    optimizer_encoder.zero_grad()
    optimizer_domain.zero_grad()

    #evaluate encoder
    questions_batch = encoder.evaluate(questions_title_batch, questions_body_batch)
    similar_questions_batch = encoder.evaluate(similar_questions_title_batch, similar_questions_body_batch)
    negative_questions_batch = evaluate_multi_questions(encoder, negative_questions_title_batch, negative_questions_body_batch)

    ubuntu_questions_batch = evaluate_multi_questions(encoder, ubuntu_rand_questions_title_batch, ubuntu_rand_questions_body_batch)
    android_questions_batch = evaluate_multi_questions(encoder, android_rand_questions_title_batch, android_rand_questions_body_batch)

    #evaluate classifier
    ubuntu_labels_probabilities = torch.cat([classifier(ubuntu_questions_batch[:,:,i]) for i in xrange(ubuntu_questions_batch.data.shape[2])])
    android_labels_probabilities = torch.cat([classifier(ubuntu_questions_batch[:,:,i])for i in xrange(android_questions_batch.data.shape[2])])

    #get loss
    loss = loss_fn(questions_batch, similar_questions_batch, negative_questions_batch, ubuntu_labels_probabilities, android_labels_probabilities)
    #optimize params
    loss.backward()
    optimizer_encoder.step()
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
    n_epochs = 20
    batch_size = 16

    encoder = CNN(feature_vector_dimensions, questions_vector_dimensions, kernel_size).cuda()
    classifier = FFNN(questions_vector_dimensions, classifier_hidden_size_1, classifier_hidden_size_2, num_labels).cuda()

    lamb = 1e-3

    database = TransferLearningDatabase()

    training_dataset = database.get_training_set()
    validation_dataset = database.get_validation_set()
    test_dataset = database.get_testing_set()

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_domain = torch.optim.Adam(classifier.parameters(), lr=-learning_rate, weight_decay=weight_decay)

    for epoch in xrange(n_epochs):
        train_epoch(encoder, classifier, training_dataset, optimizer_encoder, optimizer_domain, batch_size, lamb)
        print test_auc(encoder, validation_dataset)

    print test_auc(encoder, test_dataset)
