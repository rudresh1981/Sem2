# --------------------------
# IMPORT LIBRARIES
# --------------------------
import streamlit as st
import re
import time
from collections import defaultdict
import pandas as pd

# NLP libraries
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.metrics import edit_distance
import nltk

# Download required data (only first run)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# --------------------------
# INITIAL SETUP
# --------------------------
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --------------------------
# TEXT PROCESSING
# --------------------------

def tokenize(text):
    """Convert text into words"""
    return re.findall(r'\b\w+\b', text.lower())


def preprocess(text, mode="none"):
    """Apply preprocessing (none / stem / lemma)"""
    tokens = tokenize(text)

    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]

    # Apply stemming
    if mode == "stem":
        tokens = [stemmer.stem(t) for t in tokens]

    # Apply lemmatization
    elif mode == "lemma":
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens

# --------------------------
# INDEXING
# --------------------------

def inverted_index(docs, mode="none"):
    """Build inverted index"""
    index = defaultdict(set)

    for doc_id, doc in enumerate(docs):
        tokens = preprocess(doc, mode)

        for word in tokens:
            index[word].add(doc_id)

    return index


def positional_index(docs):
    """Build positional index"""
    index = defaultdict(lambda: defaultdict(list))

    for doc_id, doc in enumerate(docs):
        tokens = tokenize(doc)

        for pos, word in enumerate(tokens):
            index[word][doc_id].append(pos)

    return index


def biword_index(docs):
    """Build biword index"""
    index = defaultdict(set)

    for doc_id, doc in enumerate(docs):
        tokens = tokenize(doc)

        for i in range(len(tokens) - 1):
            pair = tokens[i] + " " + tokens[i+1]
            index[pair].add(doc_id)

    return index

# --------------------------
# SEARCH FUNCTIONS
# --------------------------

def basic_search(query, inv_index, mode):
    tokens = preprocess(query, mode)

    results = set()
    for t in tokens:
        results |= inv_index.get(t, set())

    return results, tokens


def biword_search(query, index):
    return list(index.get(query.lower(), []))


def positional_search(query, pos_index):
    words = query.lower().split()
    result = []

    if words[0] not in pos_index:
        return []

    for doc in pos_index[words[0]]:
        positions = pos_index[words[0]][doc]

        for p in positions:
            match = True

            for i in range(1, len(words)):
                if doc not in pos_index[words[i]] or \
                   (p + i) not in pos_index[words[i]][doc]:
                    match = False
                    break

            if match:
                result.append(doc)
                break

    return result

# --------------------------
# BST IMPLEMENTATION
# --------------------------

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def bst_insert(root, val):
    if root is None:
        return Node(val)
    if val < root.val:
        root.left = bst_insert(root.left, val)
    else:
        root.right = bst_insert(root.right, val)
    return root


def bst_search(root, val):
    if root is None:
        return False
    if root.val == val:
        return True
    elif val < root.val:
        return bst_search(root.left, val)
    return bst_search(root.right, val)

# --------------------------
# SIMPLIFIED B-TREE
# --------------------------

class BTree:
    def __init__(self):
        self.data = []

    def insert(self, key):
        self.data.append(key)
        self.data.sort()

    def search(self, key):
        low, high = 0, len(self.data) - 1

        while low <= high:
            mid = (low + high) // 2

            if self.data[mid] == key:
                return True
            elif key < self.data[mid]:
                high = mid - 1
            else:
                low = mid + 1

        return False

# --------------------------
# SPELL CORRECTION
# --------------------------

def spell_correct(word, vocab):
    best = None
    min_dist = float('inf')

    for v in vocab:
        dist = edit_distance(word, v)
        if dist < min_dist:
            min_dist = dist
            best = v

    return best

# --------------------------
# STREAMLIT APP
# --------------------------

st.title("Information Retrieval System")

file = st.file_uploader("Upload dataset (.txt)", type=["txt"])

if file:
    docs = file.read().decode("utf-8").split("\n")

    # --------------------------
    # SHOW DOCUMENTS
    # --------------------------
    st.subheader("Documents")
    st.write(docs)

    # --------------------------
    # PREPROCESSING TABLE
    # --------------------------
    st.subheader("Preprocessing Comparison")

    sample = "The machine-learning models are running efficiently"
    tokens = tokenize(sample)
    lower = [t.lower() for t in tokens]
    no_stop = [t for t in lower if t not in stop_words]
    hyphen = tokenize(sample.replace("-", " "))
    stem = [stemmer.stem(t) for t in no_stop]
    lemma = [lemmatizer.lemmatize(t) for t in no_stop]

    df = pd.DataFrame({
        "Step": ["Original", "Tokenize", "Lowercase",
                 "Stopword Removal", "Hyphen Handling",
                 "Stemming", "Lemmatization"],
        "Output": [
            sample,
            ", ".join(tokens),
            ", ".join(lower),
            ", ".join(no_stop),
            ", ".join(hyphen),
            ", ".join(stem),
            ", ".join(lemma)
        ]
    })

    st.dataframe(df)

    # --------------------------
    # SELECT MODE
    # --------------------------
    mode = st.selectbox("Select Preprocessing", ["none", "stem", "lemma"])

    inv = inverted_index(docs, mode)
    pos = positional_index(docs)
    bi = biword_index(docs)

    st.write("Vocabulary size:", len(inv))

    # --------------------------
    # QUERY INPUT
    # --------------------------
    query = st.text_input("Enter Query")

    if query:

        # BASIC SEARCH
        st.subheader("Basic Search")
        results, tokens = basic_search(query, inv, mode)

        for d in results:
            st.write(f"Doc {d}: {docs[d]}")

        # PHRASE QUERY
        st.subheader(" Phrase Query")

        st.write("Biword:")
        for d in biword_search(query, bi):
            st.write(f"Doc {d}: {docs[d]}")

        st.write("Positional:")
        for d in positional_search(query, pos):
            st.write(f"Doc {d}: {docs[d]}")

        # --------------------------
        # BST vs B-TREE TIMING
        # --------------------------
        st.subheader(" Performance Comparison")

        vocab = list(inv.keys())

        root = None
        for v in vocab:
            root = bst_insert(root, v)

        btree = BTree()
        for v in vocab:
            btree.insert(v)

        rows = []

        for t in tokens:
            start = time.perf_counter()
            bst_search(root, t)
            bst_time = (time.perf_counter() - start) * 1e6

            start = time.perf_counter()
            btree.search(t)
            bt_time = (time.perf_counter() - start) * 1e6

            rows.append({
                "Token": t,
                "BST (µs)": round(bst_time, 4),
                "B-Tree (µs)": round(bt_time, 4)
            })

        st.dataframe(pd.DataFrame(rows))

        # --------------------------
        # TOLERANT RETRIEVAL
        # --------------------------
        st.subheader("Tolerant Retrieval")

        corrected = spell_correct(query, vocab)
        st.write("Corrected Query:", corrected)

        for d in inv.get(corrected, []):
            st.write(f"Doc {d}: {docs[d]}")



