from database import QueryDatabase, UbuntuDatabase, AndroidDatabase
import metrics
import pprint
from sklearn.metrics.pairwise import paired_cosine_distances as cosine
from tqdm import tqdm
import torch
from torch.utils import data

def compute_baselines_part1():
    query_database = QueryDatabase()
    validation_set = query_database.get_validation_dataset()
    testing_set = query_database.get_testing_dataset()

    metrics_list = []
    for dataset in (validation_set, testing_set):
        similarity_vector_list = []
        bm25_scores_list = []
        for ind_sample in dataset:
            similarity_vector_list.append(ind_sample["similarity_vec"].numpy())
            bm25_scores_list.append(ind_sample["bm25_scores"].numpy())
        metrics_list.append(metrics.compute_metrics(bm25_scores_list, similarity_vector_list))
    return {"validation": metrics_list[0], "testing": metrics_list[1]}


def compute_baselines_part2():
    android_database = AndroidDatabase(use_count_vectorizer=True)
    validation_set = android_database.get_validation_dataset()
    testing_set = android_database.get_testing_dataset()

    metrics_list = []

    for dataset in (validation_set, testing_set):
        meter = metrics.AUCMeter()
        for query_pair in tqdm(dataset):
            question1_vec = (query_pair["id_1_title_vec"] + query_pair["id_1_body_vec"]) / 2.0
            question2_vec = (query_pair["id_2_title_vec"] + query_pair["id_2_body_vec"]) / 2.0
            score = (1 - cosine(question1_vec.numpy(), question2_vec.numpy()))[0]
            similarity = query_pair["similarity"]

            meter.add(torch.FloatTensor([score]), torch.LongTensor([similarity]))
        metrics_list.append(meter.value(0.05))

    return {"validation": metrics_list[0], "testing": metrics_list[1]}


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(compute_baselines_part2())