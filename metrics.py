from sklearn.metrics.ranking import label_ranking_average_precision_score as get_map_sklearn
import numpy as np
MAP = 'MAP'
MRR = 'MRR'
P1 = 'P1'
P5 = 'P5'

import numbers
import numpy as np
import torch

class Meter(object):
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class AUCMeter(Meter):
    """
    The AUCMeter measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems. The area under the curve (AUC)
    can be interpreted as the probability that, given a randomly selected positive
    example and a randomly selected negative example, the positive example is
    assigned a higher score by the classification model than the negative example.

    The AUCMeter is designed to operate on one-dimensional Tensors `output`
    and `target`, where (1) the `output` contains model output scores that ought to
    be higher when the model is more convinced that the example should be positively
    labeled, and smaller when the model believes the example should be negatively
    labeled (for instance, the output of a signoid function); and (2) the `target`
    contains only values 0 (for negative examples) and 1 (for positive examples).
    """
    def __init__(self):
        super(AUCMeter, self).__init__()
        self.reset()

    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        assert np.ndim(output) == 1, \
            'wrong output size (1D expected)'
        assert np.ndim(target) == 1, \
            'wrong target size (1D expected)'
        assert output.shape[0] == target.shape[0], \
            'number of outputs and targets does not match'
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
            'targets should be binary (0, 1)'

        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)
        self.sortind = None


    def value(self, max_fpr=1.0):
        assert max_fpr > 0

        # case when number of elements added are 0
        if self.scores.shape[0] == 0:
            return 0.5

        # sorting the arrays
        if self.sortind is None:
            scores, sortind = torch.sort(torch.from_numpy(self.scores), dim=0, descending=True)
            scores = scores.numpy()
            self.sortind = sortind.numpy()
        else:
            scores, sortind = self.scores, self.sortind

        # creating the roc curve
        tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            if self.targets[sortind[i - 1]] == 1:
                tpr[i] = tpr[i - 1] + 1
                fpr[i] = fpr[i - 1]
            else:
                tpr[i] = tpr[i - 1]
                fpr[i] = fpr[i - 1] + 1

        tpr /= (self.targets.sum() * 1.0)
        fpr /= ((self.targets - 1.0).sum() * -1.0)

        for n in range(1, scores.size + 1):
            if fpr[n] >= max_fpr:
                break

        # calculating area under curve using trapezoidal rule
        #n = tpr.shape[0]
        h = fpr[1:n] - fpr[0:n - 1]
        sum_h = np.zeros(fpr.shape)
        sum_h[0:n - 1] = h
        sum_h[1:n] += h
        area = (sum_h * tpr).sum() / 2.0

        return area / max_fpr

def compute_metrics(cosines_list, similarity_vector_list):
    metrics = {}
    get_p1 = lambda x, y: get_pk(x, y, 1)
    get_p5 = lambda x, y: get_pk(x, y, 5)
    metrics[MAP] = round(get_map_sklearn(similarity_vector_list, cosines_list), 3)
    metrics[MRR] = compute_average_metrics(get_mrr, cosines_list, similarity_vector_list)
    metrics[P1] = compute_average_metrics(get_p1, cosines_list, similarity_vector_list)
    metrics[P5] = compute_average_metrics(get_p5, cosines_list, similarity_vector_list)
    return metrics


def compute_average_metrics(metrics_function, cosines_list, similarity_vector_list):
    performances = filter(lambda x: x is not None, map(metrics_function, cosines_list, similarity_vector_list))
    return round(sum(performances) / float(len(performances)), 3)


def get_mrr(cosines, similarity_vector):
    cosines_indices = map(lambda i: (cosines[i], i), range(len(cosines)))
    cosines_indices.sort()
    cosines_indices.reverse()
    for rank, (cosine, i) in enumerate(cosines_indices):
        if similarity_vector[i] == 1:
            return 1.0 / (rank+1)
    return None


def get_pk(cosines, similarity_vector, k):
    max_ranks = np.argsort(cosines)[-k:]
    return np.average(similarity_vector[max_ranks])