import torch
from torch import nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)

    def forward(self, feature_vectors, hidden, state):
        """
        Applies this LSTM to a batch of feature vectors given a batch of hidden layers and
        states
        :param feature_vectors: A Variable wrapping a torch object of dimensions:
                (1, batch_size, input_size)
        :param hidden: A Variable wrapping a torch object of dimensions:
                (1, batch_size, hidden_size)
        :param state: A Variable wrapping a torch object of the same dimensions as input hidden
        :return: A tuple containing new_hidden and new_state, whose dimensions are the same
                as the input hidden
        """
        output, new_hidden_and_state = self.rnn(feature_vectors, (hidden, state))
        new_hidden, new_state = new_hidden_and_state
        return new_hidden, new_state

    def init_hidden(self):
        return Variable(torch.zeros(1, self.batch_size, self.hidden_size))

    def init_state(self):
        return Variable(torch.zeros(1, self.batch_size, self.hidden_size))


if __name__ == "__main__":
    input_size = 10
    hidden_size = 5
    batch_size = 3

    lstm = LSTM(input_size, hidden_size, batch_size)
    feature_vectors = Variable(torch.rand(1, batch_size, input_size))
    hidden = lstm.init_hidden()
    state = lstm.init_state()

    new_hidden, new_state = lstm(feature_vectors, hidden, state)
    print feature_vectors
    print new_hidden
    print new_state
