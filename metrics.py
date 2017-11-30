from sklearn.metrics.ranking import label_ranking_average_precision_score as get_map_sklearn
import numpy as np
MAP = 'MAP'
MRR = 'MRR'
P1 = 'P1'
P5 = 'P5'


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