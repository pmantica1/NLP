from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from itertools import product
from torch.autograd import Variable


def get_word_embeddings(filename):
    with open(filename) as file:
        embeddings = {}
        for line in file:
            line = line.rstrip().split()
            word = line[0]
            vector = [float(num) for num in line[1:]]
            embeddings[word] = np.array(vector)

        return embeddings


def get_excerpts_ratings(filename, embeddings):
    with open(filename) as file:
        excerpts = []
        ratings = []
        tokenizer = CountVectorizer().build_tokenizer()
        for line in file:
            tokens = []
            for token in tokenizer(line[1:]):
                if token in embeddings:
                    tokens.append(token)
            if tokens:
                excerpt = np.zeros(embeddings[tokens[0]].shape)
                for word in tokens:
                    excerpt += embeddings[word]
                excerpt /= np.float32(len(tokens))
                excerpts.append(torch.from_numpy(excerpt).float())

                rating = int(line[0])
                ratings.append(torch.LongTensor([rating]))

        return excerpts, ratings


class ExcerptsDataSet(data.Dataset):
    def __init__(self, excerpts, ratings):
        self.excerpts = excerpts
        self.ratings = ratings
        assert len(ratings) == len(excerpts)

    def __len__(self):
        return len(self.excerpts)

    def __getitem__(self, item):
        if item > len(self):
            raise AttributeError("index out of bounds")
        return {'excerpts': self.excerpts[item], 'ratings':self.ratings[item]}


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()

        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.input_to_output = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax()

    def forward(self, inp):
        hidden = self.input_to_hidden(inp)
        hidden = self.tanh(hidden)
        output = self.input_to_output(hidden)
        output = self.softmax(output)
        return output


def get_rating_from_output(output):
    return 0 if output.data.numpy()[0] > output.data.numpy()[1] else 1


def train_step(ffnn, batch, optimizer):
    loss_fn = nn.NLLLoss()
    excerpts = batch['excerpts']
    ratings = batch['ratings']

    def closure():
        optimizer.zero_grad()
        output = ffnn(Variable(excerpts))
        loss = loss_fn(output, Variable(ratings[:,0]))
        loss.backward()
        return loss

    optimizer.step(closure)


def train(ffnn, excerpts, ratings, learning_rate, l2_weight_decay, n_epochs, batch_size):
    optimizer = torch.optim.Adam(ffnn.parameters(), lr=learning_rate, weight_decay=l2_weight_decay)
    dataset = ExcerptsDataSet(excerpts, ratings)
    for epoch in range(n_epochs):
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        for batch in data_loader:
            train_step(ffnn, batch, optimizer)


def get_accuracy(ffnn, excerpts, ratings):
    n = len(excerpts)
    correct = 0
    for i in range(n):
        output = ffnn(Variable(excerpts[i]))
        if ratings[i].numpy()[0] == get_rating_from_output(output):
            correct += 1

    return float(correct) / n


def plot_performance(params, scores, title, xlabel, ylabel):
    plt.figure(title)
    plt.title(title)
    plt.plot(params, scores, alpha=0.5)
    plt.scatter(params, scores, c="red", marker="x")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)


if __name__ == "__main__":
    WORD_EMBEDDINGS_FILE = "../word_vectors.txt"
    TRAIN_FILE = "../data/stsa.binary.train"
    DEV_FILE = "../data/stsa.binary.dev"
    TEST_FILE = "../data/stsa.binary.test"

    embeddings = get_word_embeddings(WORD_EMBEDDINGS_FILE)
    train_excerpts, train_ratings = get_excerpts_ratings(TRAIN_FILE, embeddings)
    dev_excerpts, dev_ratings = get_excerpts_ratings(DEV_FILE, embeddings)
    test_excerpts, test_ratings = get_excerpts_ratings(TEST_FILE, embeddings)

    input_size = len(train_excerpts[0])
    output_size = 2
    n_epochs = 50
    batch_size = 100

    hidden_sizes = [input_size/4, input_size/2, input_size, 2*input_size]
    learning_rates = [1e-5, 1e-3, 1e-1, 1e1]
    l2_weight_decays = [1e-5, 1e-3, 1e1]

    ######################
    #Submitted Code Block#
    ######################

    best_hidden_size = 2*input_size
    best_learning_rate = 1e-3
    best_l2_weight_decay = 1e-5

    torch.manual_seed(1)
    best_ffnn = FFNN(input_size, best_hidden_size, output_size)
    train(best_ffnn, train_excerpts, train_ratings, best_learning_rate, best_l2_weight_decay, n_epochs, batch_size)
    best_dev_score = get_accuracy(best_ffnn, dev_excerpts, dev_ratings)

    print "Best Dev Performance with d={}, lr={}, weight_decay={} was: {}".format(\
        str(best_hidden_size), str(best_learning_rate), str(best_l2_weight_decay), str(best_dev_score))

    test_score = get_accuracy(best_ffnn, test_excerpts, test_ratings)
    print "Test performace with best parameters: {}".format(str(test_score))


    ################
    #4.1 Code Block#
    ################
    '''
    parameters = product(hidden_sizes, learning_rates, l2_weight_decays)
    best_ffnn = None
    best_dev_score = 0
    best_parameters = None

    for hidden_size, learning_rate, l2_weight_decay in parameters:
        torch.manual_seed(1)
        ffnn = FFNN(input_size, hidden_size, output_size)
        train(ffnn, train_excerpts, train_ratings, learning_rate, l2_weight_decay, n_epochs, batch_size)
        dev_score = get_accuracy(ffnn, dev_excerpts, dev_ratings)
        print (dev_score, hidden_size, learning_rate, l2_weight_decay)
        if dev_score > best_dev_score:
            best_ffnn = ffnn
            best_dev_score = dev_score
            best_parameters = (hidden_size, learning_rate, l2_weight_decay)

    best_hidden_size, best_learning_rate, best_l2_weight_decay = best_parameters
    print "Best Dev Performance with d={}, lr={}, weight_decay={} was: {}".format(\
        str(best_hidden_size), str(best_learning_rate), str(best_l2_weight_decay), str(best_dev_score))

    test_score = get_accuracy(best_ffnn, test_excerpts, test_ratings)
    print "Test performace with best parameters: {}".format(str(test_score))
    '''

    ################
    #4.2 Code Block#
    ################
    '''
    import matplotlib.pyplot as plt
    best_hidden_size = 2*input_size
    best_learning_rate = 1e-3
    best_l2_weight_decay = 1e-5
    parameters = [product(hidden_sizes, [best_learning_rate], [best_l2_weight_decay]),
                  product([best_hidden_size], learning_rates, [best_l2_weight_decay]),
                  product([best_hidden_size], [best_learning_rate], l2_weight_decays)]

    scores = [[] for i in range(len(parameters))]
    for i in range(len(scores)):
        for hidden_size, learning_rate, l2_weight_decay in parameters[i]:
            torch.manual_seed(1)
            ffnn = FFNN(input_size, hidden_size, output_size)
            train(ffnn, train_excerpts, train_ratings, learning_rate, l2_weight_decay, n_epochs, batch_size)
            dev_score = get_accuracy(ffnn, dev_excerpts, dev_ratings)
            scores[i].append(dev_score)

    plot_performance(hidden_sizes, scores[0], "Variable Hidden Size", "Hidden Size", "Dev Accuracy")
    plot_performance(np.log10(learning_rates), scores[1], "Variable Learning Rate", "Log Base 10 Learning Rate", "Dev Accuracy")
    plot_performance(np.log10(l2_weight_decays), scores[2], "Variable Weight Decay", "Log Base 10 Weight Decay", "Dev Accuracy")

    plt.show()
    '''




