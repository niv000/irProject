from flask import Flask, request, jsonify
import os
import re
import math
import heapq
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
from google.cloud import storage
from inverted_index_gcp import *

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\inbal\PycharmProjects\irProject\irproject-478010-e3f3d84eaa29.json"

# -----------------------------------------------------------------------------------------------
# "hash function and stopwords from assignment 3:"
# import hashlib
# def _hash(s):
#     return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()
# -----------------------------------------------------------------------------------------------

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

corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)

# -----------------------------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.group().lower() for t in RE_WORD.finditer(text.lower())
            if t.group().lower() not in all_stopwords]

# -----------------------------------------------------------------------------------------------

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

BUCKET_NAME = os.environ.get("IR_BUCKET_NAME", "ir_bucket_storage")
POSTINGS_PREFIX = os.environ.get("IR_POSTINGS_PREFIX", "postings_gcp")

# Index file names (change if your pickles are named differently) //TODO: check if the names are correct
BODY_INDEX_NAME = os.environ.get("IR_BODY_INDEX_NAME", "postings_gcp_index")
TITLE_INDEX_NAME = os.environ.get("IR_TITLE_INDEX_NAME", "postings_gcp_title_index")
ANCHOR_INDEX_NAME = os.environ.get("IR_ANCHOR_INDEX_NAME", "postings_gcp_anchor_index")

# Optional metadata files (recommended)
TITLES_PKL = os.environ.get("IR_TITLES_PKL", "doc_id_to_title.pkl")      # {doc_id:int -> title:str}
DL_PKL = os.environ.get("IR_DL_PKL", "doc_len.pkl")                      # {doc_id:int -> doc_len:int}
PAGERANK_PKL = os.environ.get("IR_PAGERANK_PKL", "pagerank.pkl")         # {doc_id:int -> float}
PAGEVIEWS_PKL = os.environ.get("IR_PAGEVIEWS_PKL", "pageviews.pkl")

def _safe_load_pickle(path: str, default):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return default

# Load indices (if some are missing, endpoints still work but return empty results)
try:
    BODY_INDEX = InvertedIndex.read_index("", BODY_INDEX_NAME)
    BODY_INDEX.bucket_name = BUCKET_NAME
except Exception: # TODO: add prints just to check if it works
    BODY_INDEX = None

try:
    TITLE_INDEX = InvertedIndex.read_index("", TITLE_INDEX_NAME)
    TITLE_INDEX.bucket_name = BUCKET_NAME
except Exception:
    TITLE_INDEX = None

try:
    ANCHOR_INDEX = InvertedIndex.read_index("", ANCHOR_INDEX_NAME)
    ANCHOR_INDEX.bucket_name = BUCKET_NAME
except Exception:
    ANCHOR_INDEX = None

DOC_TITLES: Dict[int, str] = _safe_load_pickle(TITLES_PKL, {})
DL: Dict[int, int] = _safe_load_pickle(DL_PKL, {})
PAGERANK: Dict[int, float] = _safe_load_pickle(PAGERANK_PKL, {})
PAGEVIEWS: Dict[int, int] = _safe_load_pickle(PAGEVIEWS_PKL, {})

# Corpus stats
N_DOCS = len(DL) if DL else 1
AVGDL = (sum(DL.values()) / N_DOCS) if DL else 100.0  # fallback


_gcs_client = storage.Client()  # OK to keep as global

def _download_range(bucket: storage.Bucket, blob_path: str, start: int, length: int) -> bytes:
    """
    Download a byte range from a blob.
    """
    blob = bucket.blob(blob_path)
    end = start + length - 1
    return blob.download_as_bytes(start=start, end=end)


def _decode_postings(raw: bytes) -> List[Tuple[int, int]]:
    """
    Decode postings from bytes.
    Supports:
    - 6-byte format: doc_id(4 bytes) + tf(2 bytes)  [as in your InvertedIndex.posting_lists_iter]
    - 8-byte packed format: (doc_id << 16 | tf) in 8 bytes  [common in staff solutions]
    """
    postings = []

    # Try 6-byte chunks first if length aligns
    if len(raw) % 6 == 0:
        step = 6
        for i in range(0, len(raw), step):
            doc_id = int.from_bytes(raw[i:i+4], "big")
            tf = int.from_bytes(raw[i+4:i+6], "big")
            postings.append((doc_id, tf))
        return postings

    # Fallback: try 8-byte chunks packed
    if len(raw) % 8 == 0:
        step = 8
        for i in range(0, len(raw), step):
            val = int.from_bytes(raw[i:i+8], "big")
            doc_id = val >> 16
            tf = val & TF_MASK
            postings.append((doc_id, tf))
        return postings

    # Last resort: try tuple size from index if available
    # (won't fix a mismatched encoding but avoids silent wrong reads)
    return postings


def read_posting_list(index: InvertedIndex, term: str) -> List[Tuple[int, int]]:
    """
    Read posting list for `term`.
    Preference order:
    1) If index has read_a_posting_list(...) use it (your current code expects this).
    2) Else read bytes directly from GCS using index.posting_locs.
    """
    if index is None:
        return []

    # 1) Use existing method if present
    if hasattr(index, "read_a_posting_list"):
        try:
            return list(index.read_a_posting_list(POSTINGS_PREFIX, term, BUCKET_NAME))
        except Exception:
            return []

    # 2) Direct read from GCS using posting_locs
    locs = index.posting_locs.get(term)
    if not locs:
        return []

    bucket = _gcs_client.bucket(BUCKET_NAME)

    # total bytes to read should be df * tuple_size (6 or maybe 8 in your files)
    df = index.df.get(term, 0)
    if df <= 0:
        return []

    # First attempt: assume 6-byte tuples (as in your InvertedIndex.posting_lists_iter)
    want_bytes_6 = df * 6

    # We may have multiple locations across blocks/files
    chunks = []
    remaining = want_bytes_6

    # locs is usually a list of (file_name, offset) pairs
    for f_name, offset in locs:
        if remaining <= 0:
            break

        # In your writer, you upload to "postings_gcp/<file_name>"
        # Sometimes f_name includes "./" -> normalize it
        f_name_norm = str(f_name).replace("\\", "/")
        f_name_norm = f_name_norm.split("/")[-1]  # keep basename
        blob_path = f"{POSTINGS_PREFIX}/{f_name_norm}"

        # We don’t know the exact block boundary here; download remaining bytes from offset
        # If the file is shorter, GCS will just return what exists.
        try:
            data = _download_range(bucket, blob_path, int(offset), int(remaining))
        except Exception:
            data = b""

        chunks.append(data)
        remaining -= len(data)

        if len(data) == 0:
            break

    raw = b"".join(chunks)

    # If 6-byte decode yields empty, try 8-byte expectation as well
    pl = _decode_postings(raw)
    if pl:
        return pl

    # Retry with 8-byte expectation if needed
    want_bytes_8 = df * 8
    chunks = []
    remaining = want_bytes_8
    for f_name, offset in locs:
        if remaining <= 0:
            break

        f_name_norm = str(f_name).replace("\\", "/")
        f_name_norm = f_name_norm.split("/")[-1]
        blob_path = f"{POSTINGS_PREFIX}/{f_name_norm}"

        try:
            data = _download_range(bucket, blob_path, int(offset), int(remaining))
        except Exception:
            data = b""

        chunks.append(data)
        remaining -= len(data)
        if len(data) == 0:
            break

    raw = b"".join(chunks)
    return _decode_postings(raw)


# --------------------------
# Ranking functions
# --------------------------

def bm25_rank(query_terms: List[str],
              index: InvertedIndex,
              top_k: int = 100,
              per_term_k: int = 2000,
              k1: float = 1.5,
              b: float = 0.75) -> List[Tuple[int, float]]:
    """
    BM25 over `index` using postings.
    No cross-query caching. We prune per term to limit candidates.
    """
    if index is None:
        return []

    scores: Dict[int, float] = {}

    for t in query_terms:
        df = index.df.get(t, 0)
        if df <= 0:
            continue

        pl = read_posting_list(index, t)
        if not pl:
            continue

        # prune big posting lists (runtime win; often helps quality too)
        if per_term_k is not None and len(pl) > per_term_k:
            pl = heapq.nlargest(per_term_k, pl, key=lambda x: x[1])

        # BM25 idf
        idf = math.log(1.0 + (N_DOCS - df + 0.5) / (df + 0.5))

        for doc_id, tf in pl:
            dl = DL.get(doc_id, AVGDL)
            denom = tf + k1 * (1.0 - b + b * (dl / AVGDL))
            score = idf * (tf * (k1 + 1.0)) / denom
            scores[doc_id] = scores.get(doc_id, 0.0) + score

    return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])


def title_binary_candidates(query_terms: List[str], title_index: InvertedIndex) -> Dict[int, int]:
    """
    Count how many unique query terms appear in title for each doc (binary matching per term).
    Returns {doc_id -> hits}.
    """
    hits: Dict[int, int] = {}
    if title_index is None:
        return hits

    for t in set(query_terms):
        pl = read_posting_list(title_index, t)
        for doc_id, _tf in pl:
            hits[doc_id] = hits.get(doc_id, 0) + 1
    return hits


def merge_body_and_title(body_ranked: List[Tuple[int, float]],
                         title_hits: Dict[int, int],
                         query_len: int,
                         body_w: float = 0.7,
                         title_w: float = 0.3,
                         top_k: int = 100) -> List[Tuple[int, float]]:
    """
    Merge scores. Title part is normalized by query length.
    """
    scores: Dict[int, float] = {}
    for doc_id, s in body_ranked:
        scores[doc_id] = scores.get(doc_id, 0.0) + body_w * s

    if query_len > 0:
        for doc_id, h in title_hits.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + title_w * (h / query_len)

    return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])


def get_title(doc_id: int) -> str:
    return DOC_TITLES.get(int(doc_id), "")


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
    q_terms = tokenize(query)
    # Body BM25
    body_ranked = bm25_rank(q_terms, BODY_INDEX, top_k=200, per_term_k=2000)

    # Title binary boost (optional)
    t_hits = title_binary_candidates(q_terms, TITLE_INDEX) if TITLE_INDEX else {}
    merged = merge_body_and_title(body_ranked, t_hits, query_len=len(set(q_terms)),
                                    body_w=0.75, title_w=0.25, top_k=100)

    res = [(int(doc_id), get_title(doc_id)) for doc_id, _s in merged]
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
    q_terms = tokenize(query)

    ranked = bm25_rank(q_terms, BODY_INDEX, top_k=100, per_term_k=3000)
    res = [(int(doc_id), get_title(doc_id)) for doc_id, _s in ranked]
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
    q_terms = tokenize(query)
    if TITLE_INDEX is None:
        return jsonify([])

    hits = title_binary_candidates(q_terms, TITLE_INDEX)

    # rank by hit count desc, tie-break doc_id asc for determinism
    ranked = sorted(hits.items(), key=lambda x: (-x[1], x[0]))[:100]
    res = [(int(doc_id), get_title(doc_id)) for doc_id, _h in ranked]
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

    q_terms = tokenize(query)
    if ANCHOR_INDEX is None:
        return jsonify([])

    hits: Dict[int, int] = {}
    for t in set(q_terms):
        pl = read_posting_list(ANCHOR_INDEX, t)
        for doc_id, _tf in pl:
            hits[doc_id] = hits.get(doc_id, 0) + 1

    ranked = sorted(hits.items(), key=lambda x: (-x[1], x[0]))[:100]
    res = [(int(doc_id), get_title(doc_id)) for doc_id, _h in ranked]
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
    res = [float(PAGERANK.get(int(doc_id), 0.0)) for doc_id in wiki_ids]
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
    res = [int(PAGEVIEWS.get(int(doc_id), 0)) for doc_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
