import numpy as np
import time
import csv
import torch.utils.data as data
import torch 
from tqdm import tqdm



class QueryDatabase(data.Dataset):
    def __init__(self, sample=False):
        self.word2vec =  self.load_vectors(sample)
        self.queries = {}
        self.load_queries(sample)
        self.query_sets = self.load_query_sets(sample)
        self.validation_sets = self.load_testing_sets("data/dev.txt", sample)
        self.testing_sets = self.load_testing_sets("data/test.txt", sample)

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

    def add_query(self, id, title, body):
        """
        :param id: int id of this query
        :param title: list of words of title
        :param body: list of words of body
        :return: nothing
        """
        if id in self.queries:
            raise RuntimeError("Added same query to database")
        self.queries[id] = Query(title, body, self.word2vec)

    def load_vectors(self, sample=False):
        """
        :param sample: boolean true if we want only a sample of 1000 words
        :return: word-to-embedding dictionary
        """
        word2vec = {}
        print("Loading words...")
        count = 0
        with open("data/vectors_pruned_by_tests_fixed.200.txt") as infile:
            for line in tqdm(infile):
                count+=1
                parsed_line = line.split()
                word = parsed_line[0]
                embeddings = [float(num) for num in parsed_line[1:]]
                word2vec[word] = torch.FloatTensor(embeddings)
                if (sample and count>10000):
                    break
        return word2vec

    def load_queries(self, sample):
        """
        Loads the queries into the query dictionary
        :param sample: boolean true if we want only a sample of 1000 queries
        :return: nothing
        """
        print("Loading queries...")
        with open("data/texts_raw_fixed.txt") as infile:
            count = 0
            reader = csv.reader(infile, delimiter="\t")
            for row in tqdm(reader):
                count +=1
                id = int(row[0])
                title = row[1]
                body = row[2]
                self.add_query(id, title,body)
                if (sample and count > 10000):
                    break

    def load_query_sets(self, sample):
        """
        Loades the query set
        :param sample: boolean true if we want only a sample of 1000 querie sets
        :return: the of query sets in the text
        """
        query_sets = []
        print("Loading query sets...")
        with open("data/train_random.txt") as infile:
            count = 0
            reader = csv.reader(infile, delimiter="\t")
            for row in tqdm(reader):
                count += 1
                id = int(row[0])
                similar_queries = [int(question_id) for question_id in row[1].split()]
                random_queries = [int(question_id) for question_id in row[2].split()]
                query_set = QuerySet(id, similar_queries, random_queries, self.queries)
                query_sets.append(query_set)
                if (sample and count > 10000):
                    break
        return query_sets



    def load_testing_sets(self, filename, sample):
        """
        Loads the testing query set in filename 
        :param filename: the filename from which to extract the testing query set
        :param sample: boolean true if we want only a sample of 1000 querie sets
        :return: the of query sets in the text
        """
        testing_query_sets = []
        print("Loading %s...", filename)
        with open(filename) as infile:
            count = 0
            reader = csv.reader(infile, delimiter="\t")
            for row in tqdm(reader):
                count += 1
                id = int(row[0])
                similar_queries = [int(question_id) for question_id in row[1].split()]
                candidate_queries = [int(question_id) for question_id in row[2].split()]
                bm25_scores = [float(score) for score in row[3].split()]
                query_set = TestingQuerySet(id, similar_queries, candidate_queries, bm25_scores)
                testing_query_sets.append(query_set)
                if (sample and count > 10000):
                    break
        return testing_query_sets



class QuerySet(object):
    NUM_RAND_QUESTIONS_THRESHOLD = 20 
    def __init__(self, id, similar_queries, random_queries, queries):
        # Id of main query
        self.id = id
        # List of similar question ids
        self.similar_queries = similar_queries
        # List of random question ids
        self.random_queries = random_queries
        self.queries = queries

    def get_query_vector(self):
        return (self.queries[self.id].get_title_feature_vector(), self.queries[self.id].get_body_feature_vector())

    def get_similar_query_vector(self):
        rand_id = np.random.choice(self.similar_queries)
        return (self.queries[rand_id].get_title_feature_vector(), self.queries[rand_id].get_body_feature_vector())


    def get_random_query_vectors(self):
        rand_ids = np.random.choice(self.random_queries, QuerySet.NUM_RAND_QUESTIONS_THRESHOLD)
        title_vectors = torch.cat([self.queries[rand_id].get_title_feature_vector().unsqueeze(0) for rand_id in rand_ids],0)
        body_vectors = torch.cat([self.queries[rand_id].get_body_feature_vector().unsqueeze(0) for rand_id in rand_ids], 0)
        return (title_vectors, body_vectors)




class TestingQuerySet(object):
    def __init__(self, id, similar_queries, candidate_queries, bm25_scores):
        # Id of main query 
        self.id = id
        # Similar queries 
        self.similiar_queries = similar_queries
        # List of 20 candidate queries (super set of similar queries)
        self.candidate_queries = candidate_queries
        # Scores generated by some metric to indicate how well the main query and the candidate queries match
        self.bm25_scores = bm25_scores




class Query(object):
    # Titles with a length greater than 20 get their feature vectors truncated. Those with a smaller size are padded with zeroes
    MAX_TITLE_LENGTH = 20
    # Same as above
    MAX_BODY_LENGTH= 100
    TOKEN_VECTOR_SIZE = 200

    def __init__(self, title, body, word2vec):
        self.title_tokens = title.split()
        self.body_tokens = body.split()
        self.word2vec = word2vec

    def get_feature_vector(self, token_list, max_length):
        feature_vector = []
        for token_idx in range(min(len(token_list), max_length)):
            token = token_list[token_idx]
            if token in self.word2vec:
                feature_vector.append(self.word2vec[token].unsqueeze(0))
            else:
                feature_vector.append(torch.zeros(Query.TOKEN_VECTOR_SIZE).unsqueeze(0))
        # Padd the end with zeroes
        if len(token_list) < max_length:
            for i in range(len(token_list), max_length):
                feature_vector.append(torch.zeros(Query.TOKEN_VECTOR_SIZE).unsqueeze(0))
        return torch.cat(feature_vector)

    def get_title_feature_vector(self):
        return self.get_feature_vector(self.title_tokens, Query.MAX_TITLE_LENGTH)

    def get_body_feature_vector(self):
        return self.get_feature_vector(self.body_tokens, Query.MAX_BODY_LENGTH)


if __name__=="__main__":
    query_database = QueryDatabase(sample=False)
    print query_database[0]
    #print query_database.queries[1].get_title_feature_vector()