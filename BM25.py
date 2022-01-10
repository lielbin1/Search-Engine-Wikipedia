from itertools import chain
import time
import math
# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
from numpy.lib import math
import search_backend

class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, nf, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.nf = nf
        self.N = len(nf)  # need to change to len(nf)
        self.AVGDL = 482425 / self.N
        self.words, self.pls = zip(*self.index.posting_lists_iter())

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, queries, N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # add tokenizer for query
        scores_dic = {}
        words, pls = search_backend.get_posting_gen(self.index)
        for q_id, term_query in queries.items():
            idf = self.calc_idf(term_query)
            self.idf = idf
            candidate = search_backend.get_candidate_documents(term_query, self.index, words, pls)
            list_tup = []
            for doc_id in candidate:
                if doc_id in self.nf.keys():
                    list_tup.append((doc_id, self._score(term_query, doc_id)))
            list_tup = sorted(list_tup, key=lambda x: x[1])
            list_tup = list_tup[::-1]
            scores_dic[q_id] = list_tup[:N]
        return scores_dic

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.nf[doc_id][1]  # need to change to nf[doc_id][1]

        # add tokenizer for query
        for term in query:
            if term in self.index.df.keys():
                term_frequencies = dict(self.pls[self.words.index(term)])
                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score