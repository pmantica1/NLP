class QueryDatabase(object):
    def __init__(self, word2vec):
        self.queries = {}
        self.word2vec = word2vec

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


class Query(object):
    def __init__(self, title, body):
        self.title = title
        self.body = body

    def get_feature_vectors(self, word2vec):
        pass