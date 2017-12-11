import torch
from torch import nn
from torch.autograd import Variable
from nn_utils import DomainLoss

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, inp):
        """

        :param inp: 2-dimensional FloatTensor of shape (batch_size x feature_dimensions)
        :return: 2-dimensional FloatTensor of shape (batch_size x num_labels) containing probabilities of belonging to each label
        """
        hidden = self.input_to_hidden(inp)
        hidden = self.relu(hidden)
        output = self.hidden_to_output(hidden)
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
