from database import QueryDatabase
import metrics
import pprint

def compute_baselines():
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

if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(compute_baselines())