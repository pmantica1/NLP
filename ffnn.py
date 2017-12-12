import torch
from torch import nn
from torch.autograd import Variable
from nn_utils import DomainLoss

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(FFNN, self).__init__()
        self.input_to_hidden1 = nn.Linear(input_size, hidden_size_1)
        self.hidden1_to_hidden2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.hidden_to_output = nn.Linear(hidden_size_2, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, inp):
        """
        :param inp: 2-dimensional FloatTensor of shape (batch_size x feature_dimensions)
        :return: 2-dimensional FloatTensor of shape (batch_size x num_labels) containing probabilities of belonging to each label
        """
        hidden1 = self.input_to_hidden1(inp)
        hidden1 = self.relu(hidden1)
        hidden2 = self.hidden1_to_hidden2(hidden1)
        hidden2 = self.relu(hidden2)
        output = self.hidden_to_output(hidden2)
        output = self.softmax(output)
        return output


if __name__ == "__main__":
    classifier = FFNN(100, 10, 2)
    input1 = Variable(torch.rand(5, 100))
    output1 = classifier(input1)
    input2 = Variable(torch.rand(5, 100))
    output2 = classifier(input2)
    print output1
    print output2
    domain_loss = DomainLoss()
    print domain_loss(output1, output2)
