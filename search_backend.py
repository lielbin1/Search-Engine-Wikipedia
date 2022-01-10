import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
# from google.cloud import storage
import math
import pandas as pd
from contextlib import closing

import requests

from inverted_index_gcp import *
from inverted_index_gcp_title import InvertedIndexTitle
from inverted_index_gcp_anchor import InvertedIndexAnchor
from BM25 import BM25_from_index


os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
graphframes_jar = 'https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.2-s_2.12/graphframes-0.8.2-spark3.2-s_2.12.jar'
spark_jars = '/usr/local/lib/python3.7/dist-packages/pyspark/jars'


# import pyspark
# from pyspark.sql import *
# from pyspark.sql.functions import *
# from pyspark import SparkContext, SparkConf
# from pyspark.sql import SQLContext
# from pyspark.ml.feature import Tokenizer, RegexTokenizer
# from graphframes import *

# import hashlib
#


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became", "best", "the"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124
def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS


def read_posting_list(inverted, w, base_dir):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        if w not in inverted.df:
            return
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, base_dir)
        posting_list = {}
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list[doc_id] = tf
        return posting_list


######search#########
def get_posting_gen(index):
    words, pls = zip(*index.posting_lists_iter())
    return words, pls


def get_candidate_documents(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    list of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = []
    for term in np.unique(query_to_search):
        if term in words:
            current_list = (pls[words.index(term)])
            candidates += current_list
    return np.unique(candidates)


def query_to_dic(query):
    q_dic = {}
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    q_dic[1] = tokens
    return q_dic


def merge_results(title_scores, body_scores, title_weight=0.65, text_weight=0.35, N=100):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    dict_merge = {}
    for query_id in title_scores.keys():
        title_after_weight = dict([(doc_id, score * title_weight) for (doc_id, score) in title_scores[query_id]])
        body_after_weight = dict([(doc_id, score * text_weight) for (doc_id, score) in body_scores[query_id]])
        run = set(title_after_weight.keys()).union(set(body_after_weight.keys()))
        temp_dict = {}
        for id in run:
            temp_dict[id] = title_after_weight.get(id, 0) + body_after_weight.get(id, 0)

        dict_merge[query_id] = sorted(temp_dict.items(), key=lambda x: x[1], reverse=True)[:N]

    return dict_merge


def main_search_backend(query, index_body, index_title, id_title, nf):
    bm25_title = BM25_from_index(index_title, nf)
    bm25_body = BM25_from_index(index_body, nf)

    cran_txt_query_text_train = query_to_dic(query)

    bm25_queries_score_train_title = bm25_title.search(cran_txt_query_text_train)
    bm25_queries_score_train_body = bm25_body.search(cran_txt_query_text_train)

    merge_index = merge_results(bm25_queries_score_train_title, bm25_queries_score_train_body)
    res = []
    for i in merge_index[1]:
        res.append((i[0], id_title[i[0]]))
    print(res)
    return res


def count_nf(nf):
    sum_nf = 0
    for i in nf.values():
        sum_nf += i[1]
    return sum_nf


#####search body#########
def sim(q, d, index, nf, wij):
    simd_q = 0
    cosinesim = {}
    word_weight_in_query = {}
    list_of_tokens = [token.group() for token in RE_WORD.finditer(q.lower()) if token.group() not in all_stopwords]
    counter = Counter(list_of_tokens)
    wiq_wid = 0
    wiq_pow_qr = 0
    for k, v in counter.items():
        post = read_posting_list(index, k, 'postings_gcp')
        if post is not None:
            if d in post.keys():
                tf = post[d] / nf[d][1]
            else:
                tf = 0
            if k in index.df.keys():
                idf = math.log(len(nf) / index.df[k])
            else:
                idf = 0
        else:
            tf = 0
            idf = 0
        wiq_wid += (tf * idf)
        wiq_pow_qr += v ** 2
    wiq_pow_qr = wiq_pow_qr ** 0.5
    simd_q = wiq_wid / (wiq_pow_qr * (wij[d] ** 0.5))
    return (d, simd_q)


def search_body_back(q, n, index, nf, id_title, wij):
    # nf_l = pd.read_pickle('/content/nf.pkl')
    # id_title = pd.read_pickle('/content/id_title.pkl')
    posting_all_term_query = defaultdict(int)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(q.lower()) if token.group() not in all_stopwords]
    for token in list_of_tokens:
      post = read_posting_list(index, token, 'postings_gcp')
      posting_all_term_query = d_sum(posting_all_term_query, post)
    result = []
    list_score = []
    for d in posting_all_term_query.keys():
        list_score.append(sim(q, d, index, nf, wij))
    list_score.sort(key=lambda y: y[1], reverse=True)
    print(list_score)
    for i in list_score:
        result.append((i[0], id_title[i[0]]))
    return result[:n]


######search_title#########
def search_title_back(q, n, index, id_title):
    # print(q)
    list_of_pair_title = []
    # title = pd.read_pickle('/content/id_title.pkl')
    # index_title = pd.read_pickle('/content/index_title.pkl')
    # print(index_title.df)
    temp = defaultdict(int)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(q.lower())]
    # print(list_of_tokens)
    for tok in list_of_tokens:
        # print(tok)
        temp = d_sum(read_posting_list(index, tok, 'postings_gcp_title'), temp)
    temp = sorted(temp.items(), key=lambda kv: kv[1], reverse=True)
    # print(temp)
    if temp is not None:
        for tup in temp:
            # print(tup)
            list_of_pair_title.append((tup[0], id_title[tup[0]]))
    return list_of_pair_title[:n]


def d_sum(*dicts):
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)


######search_anchor#########
def search_anchor_back(q, n, index, id_title):
    list_of_pair_anchor = []
    # title = pd.read_pickle('/content/id_title.pkl')
    # print(index.df)
    temp = defaultdict(int)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(q.lower())]
    # print(list_of_tokens)
    for tok in list_of_tokens:
        post = read_posting_list(index, tok, 'postings_gcp_anchor')
        if post is not None:
            temp = d_sum(read_posting_list(index, tok, 'postings_gcp_anchor'), temp)
    temp = sorted(temp.items(), key=lambda kv: kv[1], reverse=True)
    if temp is not None:
        for tup in temp:
            list_of_pair_anchor.append((tup[0], id_title[tup[0]]))
    return list_of_pair_anchor[:n]


######search_page_rank#########
def page_rank_back(list_of_id, pr):
    # pr = spark.read.csv("/content/part-00000-669271a8-04ff-4925-bd5a-039695f867d5-c000.csv.gz")
    # requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
    result = []
    for id_ in list_of_id:
        result.append(pr.filter(pr._c0 == id_).collect()[0]._c1)
    return result


######search_page_view#########
def page_view_back(lst_of_doc_id, pv):
    """Rank pages from most viewed to least viewed using the above `wid2pv`
     counter.
  Parameters:
  -----------
    pages: An iterable list of pages as returned from `page_iter` where each
           item is an article with (id, title, body)
  Returns:
  --------
  A list of tuples
    Sorted list of articles from most viewed to least viewed article with
    article title and page views. For example:
    [('Langnes, Troms': 16), ('Langenes': 10), ('Langenes, Finnmark': 4), ...]
  """
    res = []
    for i in lst_of_doc_id:
        res.append(pv[i])
    return res
