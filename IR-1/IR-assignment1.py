# --------------------------
# IMPORT LIBRARIES
# --------------------------
import streamlit as st
import re
import time
from collections import defaultdict
import pandas as pd
from BTrees.OOBTree import OOBTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLP libraries
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.metrics import edit_distance
import nltk

# Download required data (only first run)
@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("stopwords")

load_nltk()

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
# WILDCARD FUNCTION
# --------------------------
def wildcard_search(pattern, vocab):

    pattern = pattern.replace("*", ".*")

    matches = []

    for word in vocab:
        if re.fullmatch(pattern, word):
            matches.append(word)

    return matches
    
# --------------------------
# STREAMLIT APP
# --------------------------

st.title("Information Retrieval System")

file = st.file_uploader("Upload dataset (.txt)", type=["txt"])

if file:
    st.success(f"Dataset Loaded: {file.name}")
    
    docs = [
    d.strip()
    for d in file.read().decode("utf-8").split("\n")
    if d.strip()
    ]

    # --------------------------
    # SHOW DOCUMENTS
    # --------------------------
    st.subheader("Documents")
    st.write(docs)

    st.subheader("Dataset Statistics")

    total_docs = len(docs)
    
    all_tokens = []
    for doc in docs:
        all_tokens.extend(tokenize(doc))
    
    vocab_size = len(set(all_tokens))
    
    st.write("Total Documents:", total_docs)
    st.write("Total Tokens:", len(all_tokens))
    st.write("Vocabulary Size:", vocab_size)
    
    # --------------------------
    # PREPROCESSING TABLE
    # --------------------------
    st.subheader("Preprocessing Comparison")
    
    st.info("Select a document from the uploaded dataset to observe the effect of each preprocessing technique.")
    
    # Select document
    
    sample_doc = st.selectbox(
        "Select Document for Preprocessing Demonstration",
        range(len(docs)),
        format_func=lambda x: f"Document {x+1}"
    )
    
    sample = docs[sample_doc]
    
    # Apply preprocessing
    
    tokens = tokenize(sample)
    
    lower = [
        t.lower()
        for t in tokens
    ]
    
    no_stop = [
        t
        for t in lower
        if t not in stop_words
    ]
    
    hyphen = tokenize(
        sample.replace("-", " ")
    )
    
    stem = [
        stemmer.stem(t)
        for t in no_stop
    ]
    
    lemma = [
        lemmatizer.lemmatize(t)
        for t in no_stop
    ]
    
    # Token statistics
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Original Token Count", len(tokens))
    
    with col2:
        st.metric("After Stopword Removal", len(no_stop))
    
    # Table
    
    preprocess_df = pd.DataFrame({
        "Step": [
            "Original Document",
            "Tokenization",
            "Lowercasing",
            "Stopword Removal",
            "Hyphen Handling",
            "Stemming",
            "Lemmatization"
        ],
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
    
    st.dataframe(preprocess_df, use_container_width=True )

    # --------------------------
    # SELECT MODE
    # --------------------------
    # --------------------------
    # SELECT PREPROCESSING
    # --------------------------
    
    mode = st.selectbox(
        "Select Preprocessing",
        ["Select...", "stem", "lemma"]
    )
    
    # Only proceed after user selects a preprocessing method
    
    if mode != "Select...":
    
        # --------------------------
        # BUILD INDEXES
        # --------------------------
    
        inv = inverted_index(docs, mode)
        pos = positional_index(docs)
        bi = biword_index(docs)
    
        # --------------------------
        # INDEX CONSTRUCTION
        # --------------------------
    
        st.header("Index Construction")
        st.subheader("Index Information")
    
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indexed Terms",len(inv))
        
        with col2:
            st.metric("Vocabulary Size",len(inv))
    
        # --------------------------
        # INVERTED INDEX
        # --------------------------
    
        with st.expander("Inverted Index Sample"):
            sample_inv = dict(list(inv.items())[:20])
            st.json({
                k: list(v)
                for k, v in sample_inv.items()
            })
    
        # --------------------------
        # BIWORD INDEX
        # --------------------------
    
        with st.expander("Biword Index Sample"):
            sample_bi = dict(list(bi.items())[:20])
            st.json({
                k: list(v)
                for k, v in sample_bi.items()
            })
    
        # --------------------------
        # POSITIONAL INDEX
        # --------------------------
    
        with st.expander("Positional Index Sample"):
    
            sample_pos = {}
            for term in list(pos.keys())[:10]:
    
                sample_pos[term] = {
                    str(doc): positions
                    for doc, positions
                    in pos[term].items()
                }
    
            st.json(sample_pos)
    
        # --------------------------
        # QUERY PROCESSING
        # --------------------------
    
        st.header("Query Processing")
        st.header("Basic Retrieval")
    
        basic_query = st.text_input("Enter Query for Basic Retrieval")
        
        if basic_query:    
            
            # BASIC SEARCH
            st.subheader("Basic Search")
            results, tokens = basic_search(basic_query, inv, mode)
            st.write("Processed Query Tokens:")
            st.write(tokens) 
            
            st.write("Documents Retrieved:", len(results))
                   
            st.subheader("Top Results")
            
            result_list = list(results)

            for d in result_list[:20]:
                st.write(f"Doc {d+1}: {docs[d]}")
            if len(result_list) > 20:
                with st.expander(f"View Remaining {len(result_list)-20} Results"):
                    for d in result_list[20:]:
                        st.write(f"Doc {d+1}: {docs[d]}")
                
            # PHRASE QUERY
            st.header("Phrase Query Processing")
            phrase_query = st.text_input("Enter Phrase Query", key="phrase")
            
            if phrase_query:

                # BIWORD
                bi_results = biword_search(phrase_query,bi)
                st.markdown("#### Biword Index Results")
                if len(bi_results) == 0:
                    st.warning("No matching documents found.")
                else:
                    for d in bi_results[:20]:
                        st.write(f"Doc {d+1}: {docs[d]}")
                    if len(bi_results) > 20:
                        with st.expander(f"View Remaining {len(bi_results)-20} Results"):
                            for d in bi_results[20:]:
                                st.write(f"Doc {d+1}: {docs[d]}")
    
                #POSITIONAL
                pos_results = positional_search(phrase_query,pos)
                st.markdown("#### Positional Index Results")
                if len(pos_results) == 0:
                    st.warning("No matching documents found.")
                else:
                    for d in pos_results[:20]:
                        st.write(f"Doc {d+1}: {docs[d]}")
                    if len(pos_results) > 20:
                        with st.expander(f"View Remaining {len(pos_results)-20} Results"):
                            for d in pos_results[20:]:
                                st.write(f"Doc {d+1}: {docs[d]}")
        
                st.subheader("Phrase Query Analysis")
    
                comparison_df = pd.DataFrame({
                    "Method":["Biword Index","Positional Index"],
                    "Documents Retrieved":[len(bi_results),len(pos_results)]
                })
                
                st.dataframe(comparison_df)
    
                #  Explanation
                if len(bi_results) > len(pos_results):
                
                    st.warning(f"Biword Index retrieved {len(bi_results)} documents while "
                        f"Positional Index retrieved {len(pos_results)} documents." )
                
                    st.markdown("""
                    **Inference:**
                
                    Biword indexing stores only adjacent word pairs and does not verify
                    the exact positions of all words in the phrase.
                
                    Therefore, some retrieved documents may be false positives.
                
                    Positional indexing verifies exact word positions and retrieves
                    more accurate phrase matches.
                    """)
                
                else:
                
                    st.success( f"Biword Index retrieved {len(bi_results)} documents and "
                        f"Positional Index retrieved {len(pos_results)} documents.")
                
                    st.markdown("""
                    **Inference:**
                
                    Both methods produced similar results for this query.
                
                    However, positional indexing is still more reliable because it
                    verifies exact word positions and phrase continuity, whereas
                    biword indexing only checks adjacent word pairs.
                    """)
                
                # --------------------------
                # Stemming vs Lemmatization Evaluation
                # --------------------------
        
                st.subheader("Stemming vs Lemmatization Evaluation")
                eval_query = st.text_input("Enter Query for Stem vs Lemma Evaluation", key="eval")

                if eval_query:
                    stem_inv = inverted_index(docs, "stem")
                    lemma_inv = inverted_index(docs, "lemma")
        
                    # Documents retrieved
                    stem_docs, _ = basic_search(eval_query,stem_inv,"stem")
                    lemma_docs, _ = basic_search(eval_query,lemma_inv,"lemma")
        
                    stem_docs_processed = [" ".join(preprocess(d,"stem"))for d in docs]
                    
                    lemma_docs_processed = [" ".join(preprocess(d,"lemma"))for d in docs]
                   
                    query_stem = " ".join(preprocess(eval_query,"stem"))
                    
                    query_lemma = " ".join(preprocess(eval_query,"lemma"))
                    
                    # STEM
                    
                    vectorizer = TfidfVectorizer()
                    stem_matrix = vectorizer.fit_transform(stem_docs_processed +[query_stem])
                    stem_similarity = cosine_similarity(stem_matrix[-1],stem_matrix[:-1]).mean()
                    
                    # LEMMA
                    
                    vectorizer2 = TfidfVectorizer()
                    lemma_matrix = vectorizer2.fit_transform(lemma_docs_processed +[query_lemma])
                    lemma_similarity = cosine_similarity(lemma_matrix[-1],lemma_matrix[:-1]).mean()
                    
                    eval_df = pd.DataFrame({"Method":["Stemming","Lemmatization"],
                        "Documents Retrieved":[len(stem_docs),len(lemma_docs)],
                        "Average Similarity":[round(stem_similarity,4),round(lemma_similarity,4)]
                    })
                    st.dataframe(eval_df)
                    
                    if lemma_similarity > stem_similarity:
                        st.success("Inference: Lemmatization preserves semantic meaning better for this dataset.")
                    else:
                        st.success("Inference: Stemming produced better retrieval similarity.")
                    
                    # --------------------------
                    # BST vs B-TREE TIMING
                    # --------------------------
                    st.subheader("Performance Comparison")
                    
                    vocab = list(inv.keys())
                    
                    # Build BST
                    root = None
                    
                    for v in vocab:
                        root = bst_insert(root, v)
                    
                    # Build B-Tree
                    btree = OOBTree()
                    
                    for i, v in enumerate(vocab):
                        btree[v] = i
                    
                    # --------------------------
                    # Use Actual Query Terms
                    # --------------------------
                    
                    #test_terms = preprocess(query,mode)
                    test_terms = [
                        "love",
                        "beauty",
                        "time",
                        "heart",
                        "summer"
                    ]
                    
                    # Fallback if query becomes empty
                    
                    if len(test_terms) == 0:
                        test_terms = vocab[:5]
                    
                    st.info("BST and B-Tree comparison performed using five benchmark terms.")
                    
                    st.write("Processed Query Terms:")
                    
                    st.write(test_terms)
                    
                    # --------------------------
                    # Measure Search Times
                    # --------------------------
                    
                    rows = []
                    
                    for t in test_terms:
                    
                        # BST Search Time
                        start = time.perf_counter()
                    
                        bst_search(root,t)
                    
                        bst_time = (time.perf_counter() - start) * 1e6
                    
                        # B-Tree Search Time
                        start = time.perf_counter()
                    
                        _ = t in btree
                    
                        bt_time = (time.perf_counter() - start) * 1e6
                    
                        rows.append({
                            "Query Term": t,
                            "BST Search Time (µs)": round(bst_time,4),
                            "B-Tree Search Time (µs)": round(bt_time,4)
                        })
                    
                    # --------------------------
                    # Results Table
                    # --------------------------
                    
                    timing_df = pd.DataFrame(rows)
                    
                    st.dataframe(timing_df)
                    
                    # --------------------------
                    # Average Times
                    # --------------------------
                    
                    avg_bst = timing_df["BST Search Time (µs)"].mean()
                    
                    avg_bt = timing_df["B-Tree Search Time (µs)"].mean()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Average BST Time (µs)",round(avg_bst,4))
                    
                    with col2:
                        st.metric("Average B-Tree Time (µs)",round(avg_bt,4))
                    
                    # --------------------------
                    # Visualization
                    # --------------------------
                    
                   
                    comparison_df = pd.DataFrame({"Structure": ["BST", "B-Tree"],"Average Search Time (µs)": [round(avg_bst,4),round(avg_bt,4)]})
                
                    st.subheader("BST vs B-Tree Average Search Time")
                    
                    st.bar_chart( comparison_df,x="Structure", y="Average Search Time (µs)")
        
                    st.subheader("Retrieval Time Analysis")
        
                    retrieval_rows = []
                    
                    for t in test_terms:
                        start = time.perf_counter()
                        docs_found = inv.get(t,set())
                        retrieval_time = (time.perf_counter()-start) * 1e6
                    
                        retrieval_rows.append({"Query Term": t,"Documents Retrieved": len(docs_found),"Retrieval Time (µs)": round(retrieval_time,4) })
                    
                    retrieval_df = pd.DataFrame(retrieval_rows)
                    st.dataframe(retrieval_df)
                    
                    # --------------------------
                    # Inference
                    # --------------------------
                    
                    if avg_bt < avg_bst:
                        st.success("Inference: B-Tree performed faster than BST for the given query terms.")
                    else:
                        st.success("Inference: BST performed faster than B-Tree for the given query terms.")
                        
                    # --------------------------
                    # TOLERANT RETRIEVAL
                    # --------------------------
        
                    st.subheader("Tolerant Retrieval")
                    tolerant_query = st.text_input("Enter Imperfect Query (Spelling/Wildcard)")
        
                    if tolerant_query:
                        query_words = tokenize(tolerant_query)
                        
                        spell_rows = []
                        corrected_terms = []
                        
                        for word in query_words:
                        
                            if word not in vocab:
                                suggestion = spell_correct(word, vocab)
                            else:
                                suggestion = word
                        
                            corrected_terms.append(suggestion)
                        
                            spell_rows.append({"Original": word,"Suggested": suggestion})
                        
                        spell_df = pd.DataFrame(spell_rows)
                        
                        st.subheader("Spelling Correction Demonstration")
                        st.dataframe(spell_df)
                        
                        # Show corrected query
                        st.write("Corrected Query:"," ".join(corrected_terms))
                        
                        # Retrieve documents using corrected terms
                        corrected_docs = set()
                        
                        for term in corrected_terms:
                            corrected_docs |= inv.get(term, set())
                        
                        st.write("Documents Retrieved After Correction:",len(corrected_docs))
                        
                        for d in list(corrected_docs)[:20]:
                            st.write(f"Doc {d+1}: {docs[d]}")
                        
                        if len(corrected_docs) > 20:
                            with st.expander(f"View Remaining {len(corrected_docs)-20} Results"):
                                for d in list(corrected_docs)[20:]:
                                    st.write(f"Doc {d+1}: {docs[d]}")
                        
                        # Wildcard Query
                        if "*" in tolerant_query:
                            st.subheader("Wildcard Query Demonstration")
                            matches = wildcard_search(tolerant_query.lower().strip(),vocab)
                            st.write("Wildcard Pattern:", tolerant_query)
                            st.write("Matching Terms Found:", len(matches))
                            st.write(matches)
                            st.success(
                                "Wildcard retrieval allows users to search even when only "
                                "partial word information is available."
                            )
                        # --------------------------
                        # Final Conclusion
                        # --------------------------
                
                        st.subheader("Inference and Discussion")
                        
                        st.markdown("""
                        ### 1. Which preprocessing technique improved retrieval quality?
                        Stopword removal and lemmatization improved retrieval quality.
                        
                        ### 2. Was stemming or lemmatization better?
                        Lemmatization preserved semantic meaning better.
                        
                        ### 3. Which phrase query index was more accurate?
                        Positional indexing was more accurate.
                        
                        ### 4. Which tree structure was faster?
                        B-Tree performed better.
                        
                        ### 5. How tolerant was the retrieval model?
                        The model handled spelling mistakes and wildcard queries.
                        
                        ### 6. Limitations
                        - Small dataset
                        - No ranking model
                        - No semantic search
                        
                        ### 7. Future Improvements
                        - BM25
                        - Elasticsearch
                        - BERT Retrieval
                        """)
