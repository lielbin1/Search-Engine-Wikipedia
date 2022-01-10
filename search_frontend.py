from pathlib import Path
from google.cloud import storage
from flask import Flask, request, jsonify, render_template
import search_backend
import pickle
from google.colab import auth, drive
auth.authenticate_user()
import os
import requests
#import pyspark
#from pyspark.sql import *
#from pyspark.sql.functions import *
#from pyspark import SparkContext, SparkConf
#from pyspark.sql import SQLContext
# from pyspark.ml.feature import Tokenizer, RegexTokenizer
# from graphframes import *
import math


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        # drive.mount('content/gDrive')
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)




# sc = pyspark.SparkContext(conf=conf)
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
graphframes_jar = 'https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.2-s_2.12/graphframes-0.8.2-spark3.2-s_2.12.jar'


# spark_jars = '/usr/local/lib/python3.7/dist-packages/pyspark/jars'
# conf = SparkConf().set("spark.ui.port", "4050")
# sc = SparkContext.getOrCreate(conf=conf)
# sc.addPyFile(str(Path(spark_jars) / Path(graphframes_jar).name))
# spark = SparkSession.builder.getOrCreate()

bucket_name = 'search_wiki_319081600'
client = storage.Client()
blobs = client.list_blobs(bucket_name)
bucket = client.get_bucket("search_wiki_319081600")

# pr = spark.read.csv('gs://wiki_319081600/pr/part-00000-433ea158-13df-4ceb-9af1-9e75a14f86b0-c000.csv.gz')

blob = bucket.get_blob('postings_gcp/nf.pkl')
with blob.open("rb") as f:
    nf = pickle.load(f)

blob = bucket.get_blob('id_tit/id_title.pkl')
with blob.open("rb") as f:
    id_title = pickle.load(f)

blob = bucket.get_blob('postings_gcp/index.pkl')
with blob.open("rb") as f:
    index_body = pickle.load(f)

blob = bucket.get_blob('postings_gcp_title/index_title.pkl')
with blob.open("rb") as f:
    index_title = pickle.load(f)

blob = bucket.get_blob('postings_gcp_anchor/index_anchor.pkl')
with blob.open("rb") as f:
    index_anchor = pickle.load(f)

blob = bucket.get_blob('tfidf/tfidf.pkl')
with blob.open("rb") as f:
    tfidf = pickle.load(f)

blob = bucket.get_blob('pv/pageviews-202108-user.pkl')
with blob.open("rb") as f:
    pv = pickle.load(f)



app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


#
# @app.route("/")
# def index():
#     return render_template("index.html")


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_backend.search_title_back(query, 100, index_title, id_title)
    # END SOLUTION
    print(res)
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_backend.search_body_back(query, 100, index_body, nf, id_title, tfidf)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_backend.search_title_back(query, 100, index_title, id_title)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_backend.search_anchor_back(query, 100, index_anchor, id_title)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    print('data from client:', input_json)
    dictToReturn = {'answer': 42}
    res = search_backend.page_rank_back(wiki_ids, pr)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_backend.page_view_back(wiki_ids, pv)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
