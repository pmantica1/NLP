import numpy as np
import time
import csv
from tqdm import tqdm

class QueryDatabase(object):
    def __init__(self, sample=False):
        self.word2vec =  self.load_vectors(sample)
        self.queries = {}
        self.load_queries(sample)
        self.querie_sets = self.load_query_sets(sample)

    def add_query(self, id, title, body):
        """
        :param id: int id of this query
        :param title: list of words of title
        :param body: list of words of body
        :return: nothing
        """
        if id in self.queries:
            raise RuntimeError("Added same query to database")
        self.queries[id] = Query(title, body)

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
                word2vec[word] = np.array(embeddings)
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
                similar_questions = [int(question_id) for question_id in row[1].split()]
                random_questions = [int(question_id) for question_id in row[2].split()]
                query_set = QuerySet(id, similar_questions, random_questions)
                query_sets.append(query_set)
                if (sample and count > 10000):
                    break
        return query_sets

class QuerySet(object):
    def __init__(self, id, similar_questions, random_questions):
        # Id of main query
        self.id = id
        # List of similar question ids
        self.similar_questions = similar_questions
        # List of random question ids
        self.random_questions = random_questions

class Query(object):
    # Titles with a length greater than 20 get their feature vectors truncated. Those with a smaller size are padded with zeroes
    MAX_TITLE_LENGTH = 20
    # Same as above
    MAX_BODY_LENGTH= 100
    TOKEN_VECTOR_SIZE = 200

    def __init__(self, title, body):
        self.title_tokens  = title.split()
        self.body_tokens = body.split()

    def get_feature_vector(self, word2vec, token_list, max_length):
        feature_vector = []
        for token in range(min(len(token_list), max_length)):
            if token in word2vec:
                feature_vector.append(word2vec[token])
            else:
                feature_vector.append(np.zeros(Query.TOKEN_VECTOR_SIZE))
        # Padd the end with zeroes
        if len(token_list) < max_length:
            for i in range(max_length- len(token_list)):
                feature_vector.append(np.zeros(Query.TOKEN_VECTOR_SIZE))
        return np.concatenate(feature_vector)

    def get_title_feature_vector(self, word2vec):
        return self.get_feature_vector(word2vec, self.title_tokens, Query.MAX_TITLE_LENGTH)

    def get_body_feature_vector(self, word2vec):
        return self.get_feature_vector(word2vec, self.body_tokens, Query.MAX_BODY_LENGTH)


if __name__=="__main__":
    query_database = QueryDatabase(sample=True)