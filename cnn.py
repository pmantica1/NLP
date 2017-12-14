import torch
import database 
from torch import nn
from torch.autograd import Variable
import torch.utils.data as data
from tqdm import tqdm
from scipy.spatial.distance import cosine
from metrics import compute_metrics

from nn_utils import train_epoch, test, test_auc

TITLE_VEC = "title_vec"
BODY_VEC = "body_vec"
SIM_TITLE_VEC = "sim_title_vec"
SIM_BODY_VEC = 'sim_body_vec'
RAND_TITLE_VECS = "rand_title_vecs"
RAND_BODY_VECS = "rand_body_vecs"
SIMILARITY_VEC = "similarity_vec"
BM25_SCORES = "bm25_scores"
CAND_TITLE_VECS = "cand_title_vecs"
CAND_BODY_VECS = "cand_body_vecs"
MAP = 'MAP'
MRR = 'MRR'
P1 = 'P1'
P5 = 'P5'


class CNN(nn.Module):
    def __init__(self, feature_vector_dimensions, output_size, kernel_size):
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(feature_vector_dimensions, output_size, kernel_size)
        self.tanh = nn.Tanh()

    def forward(self, feature_vectors):
        """
        Apply this cnn to a batch of feature vectors
        :param feature_vectors: A Variable wrapping a torch of dimensions:
                    (batch size) x (feature vector dimensions) x (number of feature vectors per sentence)
        :return: a Variable containing the output of dimensions:
                    (batch size) x (output_size) x 1
        """
        output = self.cnn(feature_vectors)
        output = self.tanh(output)
        output = output.mean(dim=2).unsqueeze(2)
        return output

    def evaluate(self, title, body):
        title_vec = self.forward(Variable(title.permute(0, 2, 1)).cuda())
        body_vec = self.forward(Variable(body.permute(0, 2, 1)).cuda())
        return (title_vec + body_vec) / 2


if __name__ == "__main__":
    """
    cnn = CNN(200, 20, 3)
    input = Variable(torch.rand(50, 200, 20))
    output = cnn(input)
    loss_fn = Loss()
    loss = loss_fn(output, Variable(torch.rand(50, 20, 1)), Variable(torch.rand(50, 20, 20)))
    print loss
    loss.backward()
    """
    feature_vector_dimensions = 300
    questions_vector_dimensions_list = [200, 300, 500]
    kernel_size = 3

    learning_rate_list = [1e-4, 1e-5, 1e-6]
    weight_decay_list = [1e-3, 1e-4] 
    n_epochs = 5
    batch_size = 16

    best_auc = 0 
    best_param = [0, 0, 0]
    for questions_vector_dimensions in questions_vector_dimensions_list:
        for learning_rate in learning_rate_list:
            for weight_decay in weight_decay_list:
                cnn = CNN(feature_vector_dimensions, questions_vector_dimensions, kernel_size).cuda()
            

                ubuntu_database = database.UbuntuDatabase(use_glove=True)
                android_database = database.AndroidDatabase(use_glove=True)

                training_dataset = ubuntu_database.get_training_dataset()
                validation_dataset = android_database.get_validation_dataset()
                test_dataset = android_database.get_testing_dataset()

                optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)

                for epoch in xrange(n_epochs):
                    train_epoch(cnn, training_dataset, optimizer, batch_size)
                score = test_auc(cnn, validation_dataset)
                if score > best_auc:
                    best_auc = score 
                    best_param = [questions_vector_dimensions, learning_rate, weight_decay]
                
            
        
    print(best_auc)
    print(best_param)



