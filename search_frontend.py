from flask import Flask, request, jsonify
import os
# TODO: CHECK PATH
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\inbal\PycharmProjects\irProject\irproject-478010-e3f3d84eaa29.json"
import math
import heapq

# -----------------------------------------------------------------------------------------------
"hash function and stopwords from assignment 3:"

import sys
from collections import Counter, OrderedDict
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
from google.cloud import storage

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

from inverted_index_gcp import *

# define english stopwords from nltk Oct 2025
english_stopwords = frozenset([
    "during", "as", "whom", "no", "so", "shouldn't", "she's", "were", "needn", "then", "on",
    "should've", "once", "very", "any", "they've", "it's", "it", "be", "why", "ma", "over",
    "you'll", "they", "you've", "am", "before", "shan", "nor", "she'd", "because", "been",
    "doesn't", "than", "will", "they'd", "not", "those", "had", "this", "through", "again",
    "ours", "having", "himself", "into", "i'm", "did", "hadn", "haven", "should", "above",
    "we've", "does", "now", "m", "down", "he'd", "herself", "t", "their", "hasn't", "few",
    "and", "mightn't", "some", "do", "the", "we're", "myself", "i'd", "won", "after",
    "needn't", "wasn't", "them", "don", "further", "we'll", "hasn", "haven't", "out", "where",
    "mustn't", "won't", "at", "against", "shan't", "has", "all", "s", "being", "he'll", "he",
    "its", "that", "more", "by", "who", "i've", "o", "that'll", "there", "too", "they'll",
    "own", "aren't", "other", "an", "here", "between", "hadn't", "isn't", "below", "yourselves",
    "ve", "isn", "wouldn", "d", "we", "couldn", "ain", "his", "wouldn't", "was", "didn", "what",
    "when", "i", "i'll", "with", "her", "same", "you're", "yours", "couldn't", "for", "doing",
    "each", "aren", "which", "such", "mightn", "up", "mustn", "you", "only", "most", "of", "me",
    "she", "he's", "in", "a", "if", "but", "these", "him", "hers", "both", "my", "she'll", "re",
    "weren", "yourself", "is", "until", "weren't", "to", "are", "itself", "you'd", "themselves",
    "ourselves", "just", "wasn", "have", "don't", "ll", "how", "they're", "about", "shouldn",
    "can", "our", "we'd", "from", "it'd", "under", "while", "off", "y", "doesn", "theirs",
    "didn't", "or", "your", "it'll"
])

corpus_stopwords = ['category', 'references', 'also', 'links', 'external', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)

"end of hash function  and stopwords from assignment 3"
# --------------------------------------------------------------------------------------------------


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


index = InvertedIndex.read_index("", "postings_gcp_index (3)")
index.bucket_name = 'ir_bucket_storage'

# get titles  per doc id
if os.path.exists('id_title_pairs.pkl'):
    with open('id_title_pairs.pkl', 'rb') as f:
        titles_dict = pickle.load(f)

# get docs length and corpus size
if os.path.exists('id_doc_length_pairs.pkl'):
    with open('id_doc_length_pairs.pkl', 'rb') as f:
        doc_length_dict = pickle.load(f)

N_DOCS = len(doc_length_dict)
AVGDL = sum(doc_length_dict.values()) / N_DOCS


def bm25_rank(query_terms, index, doc_length_dict, top_k=10, k1=1.5, b=0.75):
    """
    BM25 over index param using posting lists from GCP
    """
    if index is None:
        return []

    scores = {}

    for t in query_terms:
        # check if term exists in the index
        if t not in index.df:
            continue

        df = index.df[t]

        # read posting list for the specific term
        try:
            pl = index.read_a_posting_list("postings_gcp", t, "ir_bucket_storage")
        except Exception:
            continue

        if not pl:
            continue

        # BM25 IDF calculation
        idf = math.log(1.0 + (N_DOCS - df + 0.5) / (df + 0.5))

        for doc_id, tf in pl:
            dl = doc_length_dict.get(doc_id, AVGDL)

            denom = tf + k1 * (1.0 - b + b * (dl / AVGDL))
            score = idf * (tf * (k1 + 1.0)) / denom

            scores[doc_id] = scores.get(doc_id, 0.0) + score

    return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])


def title_binary_candidates(query_terms, titles_dict, candidate_docs):
    """
    Calculates title hits using the doc_id -> title dictionary
    """
    hits = {}

    # all unique query terms
    q_terms = set(query_terms)

    # get title for every relevant doc
    for doc_id in candidate_docs:
        title_text = titles_dict.get(doc_id, "")

        if not title_text:
            continue

        # tokenize title
        title_tokens = set(title_text.lower().split())
        # title_tokens = [token.group() for token in RE_WORD.finditer(title_text.lower())]

        # count hits
        match_count = len(q_terms.intersection(title_tokens))

        if match_count > 0:
            hits[doc_id] = match_count

    return hits


def merge_body_and_title(body_ranked, title_hits, query_len, body_w=0.7, title_w=0.3, top_k=10):
    """
    Merge BM25 body scores with Title binary matches
    """
    scores = {}

    # normalize body scores
    for doc_id, s in body_ranked:
        scores[doc_id] = scores.get(doc_id, 0.0) + (body_w * s)

    # boost any doc found by title hits
    if query_len > 0:
        for doc_id, h in title_hits.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + (title_w * (h / query_len))

    return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])


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

    # tokenization and removing stopwords
    tokens_with_stop_words = [token.group() for token in RE_WORD.finditer(query.lower())]
    query_tokens = []
    for token in tokens_with_stop_words:
        if token not in all_stopwords:
            query_tokens.append(token)

    # calculate BM25 score
    body_scores = bm25_rank(query_tokens, index, doc_length_dict)
    # relevant doc ids for query
    candidate_ids = [doc_id for doc_id, score in body_scores]
    # check for title hits
    title_hits = title_binary_candidates(query_tokens, titles_dict, candidate_ids)
    # merge scores with uneven weights
    final_scores = merge_body_and_title(body_scores, title_hits, len(query_tokens))
    res = [(doc_id, titles_dict.get(doc_id, "")) for doc_id, score in final_scores]

    # END SOLUTION
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

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

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

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

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

    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
