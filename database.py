import numpy as np
import time
import csv
import torch.utils.data as data
import torch
import datasets
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from random import shuffle

class AndroidDatabase():
    def __init__(self, use_count_vectorizer=False, use_glove=False):
        self.queryDatabase = QueryDatabase("android_data/corpus.txt", use_count_vectorizer, use_glove)
        self.validation_pos = self.load_data_pairs("android_data/dev.pos.txt")
        self.validation_neg = self.load_data_pairs("android_data/dev.neg.txt")
        self.test_pos = self.load_data_pairs("android_data/test.pos.txt")
        self.test_neg = self.load_data_pairs("android_data/test.neg.txt")

    def get_validation_dataset(self):
        validation_dataset = self.validation_neg+self.validation_pos
        shuffle(validation_dataset)
        return datasets.AndroidTestingDataset(validation_dataset)

    def get_testing_dataset(self):
        testing_dataset = self.test_neg+self.test_pos
        shuffle(testing_dataset)
        return datasets.AndroidTestingDataset(testing_dataset)

    def load_data_pairs(self, filename):
        query_pairs = []
        sentiment = (1 if "pos" in filename else 0)
        print("Loading pairs from "+filename)
        with open(filename) as infile:
            reader = csv.reader(infile, delimiter=" ")
            for row in tqdm(reader):
                id_1 = int(row[0])
                id_2 = int(row[1])
                pair = QueryPair(id_1, id_2, sentiment, self.queryDatabase.id_to_query)
                query_pairs.append(pair)
        return query_pairs


class UbuntuDatabase():
    def __init__(self, use_glove=False):
        self.queryDatabase = QueryDatabase("data/texts_raw_fixed.txt", False, use_glove)
        self.query_sets = self.load_query_sets()
        self.validation_sets = self.load_testing_sets("data/dev.txt")
        self.testing_sets = self.load_testing_sets("data/test.txt")

    def get_training_dataset(self):
        return datasets.UbuntuTrainingDataset(self.query_sets)

    def get_validation_dataset(self):
        return datasets.UbuntuTestingDataset(self.validation_sets)

    def get_testing_dataset(self):
        return datasets.UbuntuTestingDataset(self.testing_sets)

    def load_query_sets(self):
        """
        Loades the query set
        :return: the of query sets in the text
        """
        query_sets = []
        print("Loading query sets...")
        with open("data/train_random.txt") as infile:
            reader = csv.reader(infile, delimiter="\t")
            for row in tqdm(reader):
                id = int(row[0])
                similar_id_to_query = [int(question_id) for question_id in row[1].split()]
                random_id_to_query = [int(question_id) for question_id in row[2].split()]
                query_set = QuerySet(id, similar_id_to_query, random_id_to_query, self.queryDatabase.id_to_query)
                query_sets.append(query_set)
        return query_sets

    def load_testing_sets(self, filename):
        """
        Loads the testing query set in filename
        :param filename: the filename from which to extract the testing query set
        :return: the of query sets in the text
        """
        testing_query_sets = []
        print("Loading %s...", filename)
        with open(filename) as infile:
            reader = csv.reader(infile, delimiter="\t")
            for row in tqdm(reader):
                id = int(row[0])
                similar_id_to_query = [int(question_id) for question_id in row[1].split()]
                candidate_id_to_query = [int(question_id) for question_id in row[2].split()]
                bm25_scores = [float(score) for score in row[3].split()]
                query_set = TestingQuerySet(id, similar_id_to_query, candidate_id_to_query, bm25_scores, self.queryDatabase.id_to_query)
                testing_query_sets.append(query_set)
        return testing_query_sets


class QueryDatabase():
    def __init__(self, filename, use_count_vectorizer = False, use_glove=False):
        self.word2vec = self.load_vectors(use_glove)
        self.vectorizer = TfidfVectorizer()
        if (use_count_vectorizer):
            self.vectorizer.fit(self.corpus_text_generator(filename))
        self.id_to_query = {}
        self.load_id_to_query(filename, use_count_vectorizer)

    def load_vectors(self, use_glove):
        """
        :return: word-to-embedding dictionary
        """
        word2vec = {}
        print("Loading words...")
        vectors_file = ("embeddings/glove_prunned.txt" if use_glove else "data/vectors_pruned_by_tests_fixed.200.txt")
        with open(vectors_file) as infile:
            for line in tqdm(infile):
                parsed_line = line.split()
                word = parsed_line[0]
                embeddings = [float(num) for num in parsed_line[1:]]
                word2vec[word] = torch.FloatTensor(embeddings)
        return word2vec


    def load_id_to_query(self, filename, use_count_vectorizer):
        """
        Loads the id_to_query into the query dictionary
        :return: nothing
        """
        print("Loading id_to_query...")
        with open(filename) as infile:
            reader = csv.reader(infile, delimiter="\t")
            for row in tqdm(reader):
                id = int(row[0])
                title = row[1]
                body = row[2]
                if not use_count_vectorizer:
                    self.id_to_query[id] = Word2VecQuery(title, body, self.word2vec)
                else:
                    self.id_to_query[id] = VectorizerQuery(title, body, self.vectorizer)

    def corpus_text_generator(self, filename):
        """
        Loads the corpus (used to train the CountVectorizer
        :param filename:  the name of the file
        :return: nothing
        """
        print("Loading corpus...")
        with open(filename) as infile:
            reader = csv.reader(infile, delimiter="\t")
            for row in tqdm(reader):
                title = row[1]
                body = row[2]
                yield title
                yield body

class QueryPair(object):
    def __init__(self, id_1, id_2, similarity, id_to_query):
        self.id_1 = id_1
        self.id_2 = id_2
        self.id_to_query = id_to_query
        # Similarity can be 0 or +1 (positive pair)
        self.similarity = similarity

    def get_query_vector_1(self):
        return (self.id_to_query[self.id_1].get_title_feature_vector(), self.id_to_query[self.id_1].get_body_feature_vector())

    def get_query_vector_2(self):
        return (self.id_to_query[self.id_2].get_title_feature_vector(), self.id_to_query[self.id_2].get_body_feature_vector())

    def get_similarity(self):
        return self.similarity



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


class Word2VecQuery(object):
    # Titles with a length greater than 20 get their feature vectors truncated. Those with a smaller size are padded with zeroes
    MAX_TITLE_LENGTH = 20
    # Same as above
    MAX_BODY_LENGTH= 100
    TOKEN_VECTOR_SIZE = 200

    def __init__(self, title, body, word2vec):
        self.title_tokens = title.lower().split()
        self.body_tokens = body.lower().split()
        self.word2vec = word2vec

    def get_feature_vector(self, token_list, max_length):
        feature_vector = []
        for token_idx in range(min(len(token_list), max_length)):
            token = token_list[token_idx]
            if token in self.word2vec:
                feature_vector.append(self.word2vec[token].unsqueeze(0))
            else:
                feature_vector.append(torch.zeros(Word2VecQuery.TOKEN_VECTOR_SIZE).unsqueeze(0))
        # Padd the end with zeroes
        if len(token_list) < max_length:
            for i in range(len(token_list), max_length):
                feature_vector.append(torch.zeros(Word2VecQuery.TOKEN_VECTOR_SIZE).unsqueeze(0))
        return torch.cat(feature_vector)

    def get_title_feature_vector(self):
        return self.get_feature_vector(self.title_tokens, Word2VecQuery.MAX_TITLE_LENGTH)

    def get_body_feature_vector(self):
        return self.get_feature_vector(self.body_tokens, Word2VecQuery.MAX_BODY_LENGTH)

class VectorizerQuery(object):
    def __init__(self, title, body, vectorizer):
        self.title = title
        self.body = body
        self.vectorizer =vectorizer

    def get_feature_vector(self, text):
        return torch.from_numpy(self.vectorizer.transform([text.lower()]).todense())

    def get_title_feature_vector(self):
        return self.get_feature_vector(self.title)

    def get_body_feature_vector(self):
        return self.get_feature_vector(self.body)


if __name__=="__main__":
    database = AndroidDatabase(use_count_vectorizer=True, use_glove=True)
    testing_set  = database.get_testing_dataset()
    for batch in testing_set:
        break