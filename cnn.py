import torch
from torch import nn
from torch.autograd import Variable
from database import QueryDatabase
import torch.utils.data as data
from tqdm import tqdm

TITLE_VEC = "title_vec"
BODY_VEC = "body_vec"
SIM_TITLE_VEC = "sim_title_vec"
SIM_BODY_VEC = 'sim_body_vec'
RAND_TITLE_VECS = "rand_title_vecs"
RAND_BODY_VECS = "rand_body_vecs"

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
        output = self.pooling(output)
        output = output.mean(dim=2)
        output = output.unsqueeze(2)
        return output


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, question_batch, similar_question_batch, negative_questions_batch):
        other_questions_batch = torch.cat([similar_question_batch, negative_questions_batch], 2)
        expanded_question_batch = question_batch.expand(other_questions_batch.data.shape)
        scores = self.cosine_similarity(expanded_question_batch, other_questions_batch)
        margin = 0.01 * torch.ones(scores[:, 0].data.shape)
        margin[0] = 0
        margin = Variable(margin)
        return ((scores - (scores[:, 0] - margin).unsqueeze(1)).max(1)[0]).mean()


def train(cnn, dataset, learning_rate, l2_weight_decay, n_epochs, batch_size):
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=l2_weight_decay)
    for epoch in xrange(n_epochs):
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        for batch in tqdm(data_loader):
            train_step(cnn, batch, optimizer)


def train_step(cnn, batch, optimizer):
    loss_fn = Loss()

    questions_title_batch = batch[TITLE_VEC]
    questions_body_batch = batch[BODY_VEC]

    similar_questions_title_batch = batch[SIM_TITLE_VEC]
    similar_questions_body_batch = batch[SIM_BODY_VEC]

    negative_questions_title_batch = batch[RAND_TITLE_VECS]
    negative_questions_body_batch = batch[RAND_BODY_VECS]

    def closure():
        optimizer.zero_grad()

        questions_batch = evaluate(cnn, questions_title_batch, questions_body_batch)
        similar_questions_batch = evaluate(cnn, similar_questions_title_batch, similar_questions_body_batch)
        negative_questions_batch = evaluate_negative_questions(cnn, negative_questions_title_batch, negative_questions_body_batch)

        loss = loss_fn(questions_batch, similar_questions_batch, negative_questions_batch)

        loss.backward()
        print loss
        return loss

    optimizer.step(closure)


def evaluate(cnn, title, body):
    title_vec = cnn(Variable(title.permute(0,2,1)))
    body_vec = cnn(Variable(body.permute(0,2,1)))
    return (title_vec + body_vec) / 2


def evaluate_negative_questions(cnn, titles, bodies):
    if len(titles[0]) != len(bodies[0]):
        raise RuntimeError("titles and bodies have different batch size")
    vectors = [evaluate(cnn, titles[:,i], bodies[:,i]) for i in xrange(len(titles[0]))]
    return torch.cat(vectors, 2)


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
    feature_vector_dimensions = 200
    questions_vector_dimensions = 667
    kernel_size = 3

    learning_rate = 1e-3
    l2_weight_decay = 1e-5
    n_epochs = 2
    batch_size = 50

    cnn = CNN(feature_vector_dimensions, questions_vector_dimensions, kernel_size)

    database = QueryDatabase()
    dataset = database.get_training_dataset()

    train(cnn, dataset, learning_rate, l2_weight_decay, n_epochs, batch_size)
