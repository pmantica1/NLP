import numpy as np
import time
import csv
import torch.utils.data as data
import torch
import datasets
from tqdm import tqdm



class QueryDatabase():
    def __init__(self, sample=False):
        self.word2vec =  self.load_vectors(sample)
        self.id_to_query = {}
        self.load_id_to_query(sample)
        self.query_sets = self.load_query_sets(sample)
        self.validation_sets = self.load_testing_sets("data/dev.txt", sample)
        self.testing_sets = self.load_testing_sets("data/test.txt", sample)

    def get_training_dataset(self):
        return datasets.TrainingDataset(self.id_to_query, self.query_sets)

    def get_validation_dataset(self):
        return datasets.TestingDataset(self.id_to_query, self.validation_sets)

    def get_testing_dataset(self):
        return datasets.TestingDataset(self.id_to_query, self.testing_sets)

    def add_query(self, id, title, body):
        """
        :param id: int id of this query
        :param title: list of words of title
        :param body: list of words of body
        :return: nothing
        """
        if id in self.id_to_query:
            raise RuntimeError("Added same query to database")
        self.id_to_query[id] = Query(title, body, self.word2vec)

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

    def load_id_to_query(self, sample):
        """
        Loads the id_to_query into the query dictionary
        :param sample: boolean true if we want only a sample of 1000 id_to_query
        :return: nothing
        """
        print("Loading id_to_query...")
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
                similar_id_to_query = [int(question_id) for question_id in row[1].split()]
                random_id_to_query = [int(question_id) for question_id in row[2].split()]
                query_set = QuerySet(id, similar_id_to_query, random_id_to_query, self.id_to_query)
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
                similar_id_to_query = [int(question_id) for question_id in row[1].split()]
                candidate_id_to_query = [int(question_id) for question_id in row[2].split()]
                bm25_scores = [float(score) for score in row[3].split()]
                query_set = TestingQuerySet(id, similar_id_to_query, candidate_id_to_query, bm25_scores, self.id_to_query)
                testing_query_sets.append(query_set)
                if (sample and count > 10000):
                    break
        return testing_query_sets



class QuerySet(object):
    NUM_RAND_QUESTIONS_THRESHOLD = 20 
    def __init__(self, id, similar_id_to_query, random_id_to_query, id_to_query):
        # Id of main query
        self.id = id
        # List of similar question ids
        self.similar_id_to_query = similar_id_to_query
        # List of random question ids
        self.random_id_to_query = random_id_to_query
        self.id_to_query = id_to_query

    def get_query_vector(self):
        return (self.id_to_query[self.id].get_title_feature_vector(), self.id_to_query[self.id].get_body_feature_vector())

    def get_similar_query_vector(self):
        rand_id = np.random.choice(self.similar_id_to_query)
        return (self.id_to_query[rand_id].get_title_feature_vector(), self.id_to_query[rand_id].get_body_feature_vector())


    def get_random_query_vectors(self):
        rand_ids = np.random.choice(self.random_id_to_query, QuerySet.NUM_RAND_QUESTIONS_THRESHOLD)
        title_vectors = torch.cat([self.id_to_query[rand_id].get_title_feature_vector().unsqueeze(0) for rand_id in rand_ids],0)
        body_vectors = torch.cat([self.id_to_query[rand_id].get_body_feature_vector().unsqueeze(0) for rand_id in rand_ids], 0)
        return (title_vectors, body_vectors)


class TestingQuerySet(object):
    NUM_OF_CANDIDATE_QUERIES = 20
    def __init__(self, id, similar_queries, candidate_queries, bm25_scores, id_to_query):
        # Id of main query 
        self.id = id
        # List of 20 candidate id_to_query (super set of similar id_to_query)
        self.candidate_queries = candidate_queries
        # The similarity vector is 1 if the candidate query is similar to the original query
        self.similarity_vector = torch.LongTensor([1 if query in similar_queries else 0 for query in candidate_queries])
        # Scores generated by some metric to indicate how well the main query and the candidate id_to_query match
        self.bm25_scores = torch.FloatTensor(bm25_scores)
        self.id_to_query = id_to_query

    def get_query_vector(self):
        return (self.id_to_query[self.id].get_title_feature_vector(), self.id_to_query[self.id].get_body_feature_vector())

    def get_similarity_vector(self):
        return self.similarity_vector

    def get_candidate_query_vectors(self):
        title_vectors = torch.cat([self.id_to_query[cand_id].get_title_feature_vector().unsqueeze(0) for cand_id in self.candidate_queries], 0)
        body_vectors = torch.cat([self.id_to_query[cand_id].get_body_feature_vector().unsqueeze(0) for cand_id in self.candidate_queries], 0)
        return (title_vectors, body_vectors)

    def get_bm25_scores(self):
        return self.bm25_scores


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
    testing_set  = query_database.get_testing_dataset()
    for batch in testing_set:
        print(batch)
        break
    #print query_database