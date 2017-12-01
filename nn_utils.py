import torch
from torch import nn
from torch.autograd import Variable
from database import QueryDatabase
import torch.utils.data as data
from tqdm import tqdm
from scipy.spatial.distance import cosine
from metrics import compute_metrics


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


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, question_batch, similar_question_batch, negative_questions_batch):
        other_questions_batch = torch.cat([similar_question_batch, negative_questions_batch], 2)
        expanded_question_batch = question_batch.expand(other_questions_batch.data.shape)
        scores = self.cosine_similarity(expanded_question_batch, other_questions_batch)
        margin = 0.5 * torch.ones(scores.data.shape)
        margin[:, 0] = 0
        margin = Variable(margin)
        batch_losses = (margin + scores - scores[:, 0].unsqueeze(1).expand(scores.data.shape)).max(1)[0]
        return batch_losses.mean()


def train_epoch(nn_model, dataset, optimizer, batch_size):
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for batch in tqdm(data_loader):
        train_step(nn_model, batch, optimizer)


def train_step(nn_model, batch, optimizer):
    loss_fn = Loss()

    questions_title_batch = batch[TITLE_VEC]
    questions_body_batch = batch[BODY_VEC]

    similar_questions_title_batch = batch[SIM_TITLE_VEC]
    similar_questions_body_batch = batch[SIM_BODY_VEC]

    negative_questions_title_batch = batch[RAND_TITLE_VECS]
    negative_questions_body_batch = batch[RAND_BODY_VECS]

    def closure():
        optimizer.zero_grad()

        questions_batch = nn_model.evaluate(questions_title_batch, questions_body_batch)
        similar_questions_batch = nn_model.evaluate(similar_questions_title_batch, similar_questions_body_batch)
        negative_questions_batch = evaluate_multi_questions(nn_model, negative_questions_title_batch, negative_questions_body_batch)

        loss = loss_fn(questions_batch, similar_questions_batch, negative_questions_batch)

        loss.backward()
        return loss

    optimizer.step(closure)


def test(nn_model, dataset):
    data_loader = data.DataLoader(dataset, batch_size=1)
    cosines_list = []
    similarity_vector_list = []
    for batch in tqdm(data_loader):
        cosines, similarity_vector = test_step(nn_model, batch)
        cosines_list.append(cosines)
        similarity_vector_list.append(similarity_vector)

    print compute_metrics(cosines_list, similarity_vector_list)


def test_step(nn_model, batch):
    questions_title_batch = batch[TITLE_VEC]
    questions_body_batch = batch[BODY_VEC]

    candidate_questions_title_batch = batch[CAND_TITLE_VECS]
    candidate_questions_body_batch = batch[CAND_BODY_VECS]

    similarity_vector_batch = batch[SIMILARITY_VEC].numpy()

    question_vector_batch = nn_model.evaluate(questions_title_batch, questions_body_batch).data.numpy()
    candidate_vector_batch = evaluate_multi_questions(nn_model, candidate_questions_title_batch, candidate_questions_body_batch).data.numpy()

    question_vec = question_vector_batch[0]
    similarity_vector = similarity_vector_batch[0]
    candidate_questions_vec  = candidate_vector_batch[0]
    cosines = [1-cosine(question_vec.flatten(), candidate_questions_vec[:, i].flatten()) for i in range(len(candidate_questions_vec[0]))]
    return cosines, similarity_vector


def evaluate_multi_questions(nn_model, titles, bodies):
    if len(titles[0]) != len(bodies[0]):
        raise RuntimeError("titles and bodies have different batch size")
    vectors = [nn_model.evaluate(titles[:,i], bodies[:,i]) for i in xrange(len(titles[0]))]
    return torch.cat(vectors, 2)