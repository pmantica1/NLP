import torch
from torch import nn
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, feature_vector_dimensions, output_size, kernel_size):
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(feature_vector_dimensions, output_size, kernel_size)
        self.tanh = nn.Tanh()
        self.pooling = nn.MaxPool1d(kernel_size)

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
        return self.pooling(output)


if __name__ == "__main__":
    cnn = CNN(10, 5, 3)
    input = Variable(torch.rand(50, 10, 5))
    output = cnn(input)
    print (output)