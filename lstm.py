import torch
from torch import nn
from torch.autograd import Variable

from database import UbuntuDabatase
from nn_utils import train_epoch, test

from tqdm import tqdm
import time

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=True)

    def forward(self, feature_vectors, hidden, state):
        """
        Applies this LSTM to a batch of feature vectors given a batch of hidden layers and
        states
        :param feature_vectors: A Variable wrapping a torch object of dimensions:
                (seq_length, batch_size, input_size)
        :param hidden: A Variable wrapping a torch object of dimensions:
                (seq_length, batch_size, hidden_size)
        :param state: A Variable wrapping a torch object of the same dimensions as input hidden
        :return: A tuple containing new_hidden and new_state, whose dimensions are the same
                as the input hidden
        """
        output, new_hidden_and_state = self.rnn(feature_vectors, (hidden, state))
        new_hidden, new_state = new_hidden_and_state
        return output, new_hidden, new_state

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.hidden_size))

    def init_state(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.hidden_size))

    def evaluate(self, title, body):
        title_inp = Variable(title.permute(1, 0, 2))
        body_inp = Variable(body.permute(1, 0, 2))

        batch_size = title_inp.data.shape[1]

        title_hidden = self.init_hidden(batch_size)
        title_state = self.init_state(batch_size)

        body_hidden = self.init_hidden(batch_size)
        body_state = self.init_state(batch_size)

        title_vec, title_hidden, title_state = self.forward(title_inp, title_hidden, title_state)

        body_vec, body_hidden, body_state = self.forward(body_inp, body_hidden, body_state)

        title_vec = torch.mean(title_vec, 0).unsqueeze(0).permute(1, 2, 0)
        body_vec = torch.mean(body_vec,0).unsqueeze(0).permute(1,2,0)

        return (title_vec + body_vec) / 2

if __name__ == "__main__":
    """
    input_size = 10
    hidden_size = 5
    batch_size = 20

    lstm = LSTM(input_size, hidden_size, batch_size)
    feature_vectors = Variable(torch.rand(1, batch_size, input_size))
    hidden = lstm.init_hidden()
    state = lstm.init_state()

    new_hidden, new_state = lstm(feature_vectors, hidden, state)
    print feature_vectors
    print new_hidden
    print new_state
    loss_fn = Loss()
    loss = loss_fn(new_hidden, new_hidden, torch.rand(batch_size, hidden_size))
    print loss
    loss.backward()
    print loss
    """

    feature_vector_dimensions = 200
    questions_vector_dimensions = 200

    learning_rate = 1e-3
    weight_decay = 1e-5
    n_epochs = 20
    batch_size = 16

    lstm = LSTM(feature_vector_dimensions, questions_vector_dimensions, batch_size)

    database = UbuntuDatabase()
    training_dataset = database.get_training_dataset()
    validation_dataset = database.get_validation_dataset()
    test_dataset = database.get_testing_dataset()

    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in xrange(n_epochs):
        train_epoch(lstm, training_dataset, optimizer, batch_size)
        test(lstm, validation_dataset)

    test(lstm, test_dataset)
