import torch.utils.data as data
import np 

class TransferLearningDataset(data.Dataset):
    # Precondition is that the android queries and ubuntu queries are randomly ordered 
    def __init__(self, ubuntu_dataset, android_queries, ubuntu_queries):
        self.ubuntu_dataset = ubuntu_dataset
        self.android_queries = android_queries
        self.ubuntu_queries = ubuntu_queries

    def __len__(self):
        return len(self.ubuntu_dataset)

    def __getitem__(self, item):
        if item > len(self):
            raise AttributeError("index out of bounds")
        random_ubuntu_title_vec, random_ubuntu_body_vec = ubuntu_queries[item%len(ubuntu_queries)]
        random_android_title_vec, random_android_body_vec = android_queries[item%len(android_queries)]
        result = ubuntu_dataset[item]
        result["ubuntu_title_vec"] = random_ubuntu_title_vec
        result["ubuntu_body_vec"] = random_ubuntu_body_vec
        result["android_title_vec"] = random_android_title_vec
        result["android_body_vec"] = random_android_body_vec 
        return result 





class AndroidTestingDataset(data.Dataset):
    def __init__(self, query_pairs):
        self.query_pairs = query_pairs

    def __len__(self):
        return len(self.query_pairs)

    def __getitem__(self, item):
        if item > len(self):
            raise AttributeError("index out of bounds")
        id_1_title_vector, id_1_body_vector = self.query_pairs[item].get_query_vector_1()
        id_2_title_vector, id_2_body_vector = self.query_pairs[item].get_query_vector_2()
        similarity = self.query_pairs[item].get_similarity()


        return {"id_1_title_vec": id_1_title_vector, "id_1_body_vec": id_1_body_vector, \
                "id_2_title_vec": id_2_title_vector, "id_2_body_vec": id_2_body_vector, "similarity": similarity}



class UbuntuTrainingDataset(data.Dataset):
    def __init__(self, query_sets):
        self.query_sets = query_sets

    def __len__(self):
        return len(self.query_sets)

    def __getitem__(self, item):
        if item > len(self):
            raise AttributeError("index out of bounds")
        main_title_vector, main_body_vector = self.query_sets[item].get_query_vector()
        sim_title_vector, sim_body_vector = self.query_sets[item].get_similar_query_vector()
        random_title_vectors, random_body_vectors = self.query_sets[item].get_random_query_vectors()

        return {"title_vec": main_title_vector, "body_vec": main_body_vector,  "sim_title_vec": sim_title_vector,\
        'sim_body_vec':sim_body_vector, "rand_title_vecs": random_title_vectors, "rand_body_vecs": random_body_vectors}


class UbuntuTestingDataset(data.Dataset):
    def __init__(self, testing_sets):
        self.testing_sets = testing_sets

    def __len__(self):
        return len(self.testing_sets)

    def __getitem__(self, item):
        if item > len(self):
            raise AttributeError("index out of bounds")

        main_title_vector, main_body_vector = self.testing_sets[item].get_query_vector()
        similarity_vector = self.testing_sets[item].get_similarity_vector()
        candidate_title_vectors, candidate_body_vectors = self.testing_sets[item].get_candidate_query_vectors()
        bm25_scores = self.testing_sets[item].get_bm25_scores()

        return {"title_vec": main_title_vector, "body_vec": main_body_vector,  "similarity_vec": similarity_vector,\
        'bm25_scores':bm25_scores, "cand_title_vecs": candidate_title_vectors, "cand_body_vecs": candidate_body_vectors}


