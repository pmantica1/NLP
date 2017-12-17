import torch
from torch import nn
from torch.autograd import Variable
from database import QueryDatabase
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics.pairwise import paired_cosine_distances as cosine
from metrics import compute_metrics, AUCMeter
import numpy as np

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
ID1_TITLE_VEC = "id_1_title_vec"
ID2_TITLE_VEC = "id_2_title_vec"
ID1_BODY_VEC = "id_1_body_vec"
ID2_BODY_VEC = "id_2_body_vec"
SIMILARITY = "similarity"


class EncoderLoss(nn.Module):
    def __init__(self, margin_size=0.2):
        super(EncoderLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.margin_size = margin_size

    def forward(self, question_batch, similar_question_batch, negative_questions_batch):
        other_questions_batch = torch.cat([similar_question_batch, negative_questions_batch], 2)
        expanded_question_batch = question_batch.expand(other_questions_batch.data.shape)
        scores = self.cosine_similarity(expanded_question_batch, other_questions_batch)
        margin = self.margin_size * torch.ones(scores.data.shape)
        margin[:, 0] = 0
        margin = Variable(margin).cuda()
        batch_losses = (margin + scores - scores[:, 0].unsqueeze(1).expand(scores.data.shape)).max(1)[0]
        loss = batch_losses.mean()
        return loss


class DomainLoss(nn.Module):
    def __init__(self):
        super(DomainLoss, self).__init__()

    def forward(self, ubuntu_probabilities_batch, android_probabilities_batch):
        label_probabilities = torch.cat([ubuntu_probabilities_batch, android_probabilities_batch], dim=2)
        batch_losses = []
        for i in xrange(len(label_probabilities)):
            label_targets = Variable(torch.cat([torch.zeros(ubuntu_probabilities_batch.data.shape[2]),  torch.ones(ubuntu_probabilities_batch.data.shape[2])]).long()).cuda()
            batch_losses.append(torch.nn.functional.cross_entropy(label_probabilities[i].permute(1,0), label_targets))

        batch_losses_tensor = torch.cat(batch_losses)
        loss = batch_losses_tensor.mean()
        return loss

class AdversarialLoss(nn.Module):
    def __init__(self, lamb):
        super(AdversarialLoss, self).__init__()
        self.encoder_loss = EncoderLoss()
        self.domain_loss = DomainLoss()
        self.lamb = lamb

    def forward(self, question_batch, similar_question_batch, negative_questions_batch, ubuntu_probabilities_batch, android_probabilities_batch):
        return self.encoder_loss(question_batch, similar_question_batch, negative_questions_batch) - self.lamb * self.domain_loss(ubuntu_probabilities_batch, android_probabilities_batch), \
               self.domain_loss(ubuntu_probabilities_batch, android_probabilities_batch)


def train_epoch(nn_model, dataset, optimizer, batch_size, margin_size=0.2):
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for batch in tqdm(data_loader):
        train_step(nn_model, batch, optimizer, margin_size)


def train_step(nn_model, batch, optimizer, margin_size):
    loss_fn = EncoderLoss(margin_size)

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
    return compute_metrics(cosines_list, similarity_vector_list)


def test_step(nn_model, batch):
    questions_title_batch = batch[TITLE_VEC]
    questions_body_batch = batch[BODY_VEC]

    candidate_questions_title_batch = batch[CAND_TITLE_VECS]
    candidate_questions_body_batch = batch[CAND_BODY_VECS]

    similarity_vector_batch = batch[SIMILARITY_VEC].numpy()

    question_vector_batch = nn_model.evaluate(questions_title_batch, questions_body_batch).data.cpu().numpy()
    candidate_vector_batch = evaluate_multi_questions(nn_model, candidate_questions_title_batch, candidate_questions_body_batch).data.cpu().numpy()

    candidate_questions_vec  = candidate_vector_batch[0]
    similarity_vector = similarity_vector_batch[0]
    question_vec = question_vector_batch[0].repeat(len(candidate_questions_vec[0]), axis=1).swapaxes(1,0)

    cosines = 1 - cosine(question_vec, candidate_questions_vec.swapaxes(1, 0))
    return cosines, similarity_vector


def test_auc(nn_model, dataset):
    data_loader = data.DataLoader(dataset, batch_size=len(dataset)/100)
    meter = AUCMeter()
    for batch in tqdm(data_loader):
        score, similarity = test_auc_step(nn_model, batch)
        meter.add(score, similarity)

    return meter.value(max_fpr=0.05)


def test_auc_step(nn_model, batch):
    title1 = batch[ID1_TITLE_VEC]
    body1 = batch[ID1_BODY_VEC]

    title2 = batch[ID2_TITLE_VEC]
    body2 = batch[ID2_BODY_VEC]

    question1_vec = nn_model.evaluate(title1, body1).data.cpu().numpy()[:, :, 0]
    question2_vec = nn_model.evaluate(title2, body2).data.cpu().numpy()[:, :, 0]

    assert question1_vec.shape == question2_vec.shape
    scores = 1 - cosine(question1_vec, question2_vec)
    similarities = batch[SIMILARITY].cpu().numpy().flatten()
    return torch.FloatTensor(scores), torch.LongTensor(similarities)


def evaluate_multi_questions(nn_model, titles, bodies):
    if len(titles[0]) != len(bodies[0]):
        raise RuntimeError("titles and bodies have different batch size")
    vectors = [nn_model.evaluate(titles[:,i], bodies[:,i]) for i in xrange(len(titles[0]))]
    return torch.cat(vectors, 2)
