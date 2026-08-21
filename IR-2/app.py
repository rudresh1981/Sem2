
import re
import sqlite3
import hashlib
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

st.set_page_config(page_title="AML Intelligence IR", page_icon="🔎", layout="wide")

DB_PATH = "ir_system.db"
USER_AGENT = "BITS-IR-Assignment-StudentCrawler/1.0"
DEFAULT_SEEDS = [
    "https://www.fincen.gov/resources/advisoriesbulletinsfact-sheets",
    "https://www.fincen.gov/news",
    "https://www.fatf-gafi.org/en/publications.html",
]

# ----------------------------
# Database
# ----------------------------
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA foreign_keys = ON")
    return con

def init_db():
    con = db()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE NOT NULL,
        title TEXT,
        content TEXT NOT NULL,
        content_hash TEXT UNIQUE NOT NULL
    );
    CREATE TABLE IF NOT EXISTS metadata (
        doc_id INTEGER PRIMARY KEY,
        source_domain TEXT,
        source_url TEXT,
        depth INTEGER,
        crawled_at TEXT,
        word_count INTEGER,
        char_count INTEGER,
        content_type TEXT,
        language TEXT,
        links_count INTEGER,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS links (
        source_doc_id INTEGER,
        target_url TEXT,
        FOREIGN KEY(source_doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_meta_domain ON metadata(source_domain);
    CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_doc_id);
    """)
    con.commit()
    con.close()

init_db()

# ----------------------------
# Text processing
# ----------------------------
def clean_text(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return title[:500], text

def normalize_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()

def content_hash(text):
    return hashlib.sha256(normalize_text(text).encode("utf-8", errors="ignore")).hexdigest()

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]{2,}", text.lower())

def keyword_extract(text, n=10):
    vec = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vec.fit_transform([text])
    scores = X.toarray()[0]
    terms = vec.get_feature_names_out()
    order = np.argsort(scores)[::-1][:n]
    return [(terms[i], float(scores[i])) for i in order if scores[i] > 0]

# ----------------------------
# Crawler
# ----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def robots_allowed(url):
    try:
        p = urlparse(url)
        robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
        rp = RobotFileParser(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True

def canonical_url(url):
    url = urldefrag(url)[0].strip()
    p = urlparse(url)
    if not p.scheme:
        return ""
    return p._replace(fragment="", query=p.query).geturl().rstrip("/")

def crawl(seeds, max_depth, max_pages, stay_on_domain=True, timeout=12):
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    q = deque()
    visited = set()
    discovered = set()
    pages = []
    errors = []
    seed_domains = set()

    for seed in seeds:
        u = canonical_url(seed)
        if u:
            q.append((u, 0, u))
            discovered.add(u)
            seed_domains.add(urlparse(u).netloc.lower())

    while q and len(pages) < max_pages:
        url, depth, source_seed = q.popleft()
        if url in visited:
            continue
        visited.add(url)

        if not robots_allowed(url):
            errors.append((url, "robots.txt disallowed"))
            continue

        try:
            r = session.get(url, timeout=timeout, allow_redirects=True)
            ctype = r.headers.get("Content-Type", "")
            final_url = canonical_url(r.url)
            if r.status_code != 200:
                errors.append((url, f"HTTP {r.status_code}"))
                continue
            if "text/html" not in ctype.lower():
                continue

            title, text = clean_text(r.text)
            if len(text) < 200:
                continue

            soup = BeautifulSoup(r.text, "lxml")
            links = []
            for a in soup.find_all("a", href=True):
                target = canonical_url(urljoin(final_url, a["href"]))
                if not target or target.startswith("mailto:") or target.startswith("javascript:"):
                    continue
                tp = urlparse(target)
                if tp.scheme not in ("http", "https"):
                    continue
                if stay_on_domain and tp.netloc.lower() not in seed_domains:
                    continue
                links.append(target)

            pages.append({
                "url": final_url,
                "source_url": source_seed,
                "depth": depth,
                "title": title or final_url,
                "content": text,
                "links": sorted(set(links)),
                "content_type": ctype.split(";")[0],
            })

            if depth < max_depth:
                for target in sorted(set(links)):
                    if target not in discovered and len(discovered) < max_pages * 8:
                        discovered.add(target)
                        q.append((target, depth + 1, source_seed))
        except Exception as e:
            errors.append((url, str(e)[:200]))

    return pages, errors

def persist_pages(pages):
    con = db()
    added, dup_url, dup_doc = 0, 0, 0
    for p in pages:
        ch = content_hash(p["content"])
        try:
            cur = con.execute(
                "INSERT INTO documents(url,title,content,content_hash) VALUES(?,?,?,?)",
                (p["url"], p["title"], p["content"], ch)
            )
            doc_id = cur.lastrowid
            wc = len(tokenize(p["content"]))
            con.execute(
                """INSERT INTO metadata
                (doc_id,source_domain,source_url,depth,crawled_at,word_count,char_count,content_type,language,links_count)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (doc_id, urlparse(p["url"]).netloc, p["source_url"], p["depth"],
                 datetime.now(timezone.utc).isoformat(), wc, len(p["content"]),
                 p["content_type"], "en", len(p["links"]))
            )
            for link in p["links"]:
                con.execute("INSERT INTO links(source_doc_id,target_url) VALUES(?,?)", (doc_id, link))
            added += 1
        except sqlite3.IntegrityError:
            # Check whether URL or document content is the duplicate.
            if con.execute("SELECT 1 FROM documents WHERE url=?", (p["url"],)).fetchone():
                dup_url += 1
            elif con.execute("SELECT 1 FROM documents WHERE content_hash=?", (ch,)).fetchone():
                dup_doc += 1
    con.commit()
    con.close()
    return added, dup_url, dup_doc

# ----------------------------
# Corpus and indexing
# ----------------------------
def load_corpus():
    con = db()
    df = pd.read_sql_query("""
        SELECT d.doc_id,d.url,d.title,d.content,d.content_hash,
               m.source_domain,m.source_url,m.depth,m.crawled_at,m.word_count,
               m.char_count,m.content_type,m.language,m.links_count
        FROM documents d JOIN metadata m ON d.doc_id=m.doc_id
        ORDER BY d.doc_id
    """, con)
    con.close()
    return df

def build_vectorizer(corpus, strategy="advanced"):
    if strategy == "baseline":
        return TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b",
                               max_df=0.98, min_df=1, ngram_range=(1,1))
    return TfidfVectorizer(
        lowercase=True, stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b",
        max_df=0.98, min_df=1, ngram_range=(1,2), sublinear_tf=True
    )

def bm25_index(texts):
    tokenized = [tokenize(t) for t in texts]
    N = len(tokenized)
    avgdl = np.mean([len(x) for x in tokenized]) if N else 1
    df = Counter()
    for toks in tokenized:
        for term in set(toks):
            df[term] += 1
    idf = {term: np.log(1 + (N - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
    return tokenized, idf, avgdl

def bm25_scores(query, tokenized, idf, avgdl, k1=1.5, b=0.75):
    qterms = tokenize(query)
    scores = np.zeros(len(tokenized))
    for i, doc in enumerate(tokenized):
        dl = len(doc) or 1
        tf = Counter(doc)
        s = 0.0
        for term in qterms:
            if term not in idf:
                continue
            f = tf.get(term, 0)
            if f == 0:
                continue
            denom = f + k1 * (1 - b + b * dl / avgdl)
            s += idf[term] * (f * (k1 + 1)) / denom
        scores[i] = s
    return scores

def minmax(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.max(x) == np.min(x):
        return np.zeros_like(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def pagerank_scores(corpus):
    G = nx.DiGraph()
    ids = corpus.doc_id.tolist()
    G.add_nodes_from(ids)
    con = db()
    rows = con.execute("""
        SELECT l.source_doc_id,l.target_url,d.doc_id
        FROM links l JOIN documents d ON l.target_url=d.url
    """).fetchall()
    con.close()
    for s, target_url, t in rows:
        if s in G and t in G:
            G.add_edge(s, t)
    if len(G) == 0:
        return {i: 0.0 for i in ids}, G
    pr = nx.pagerank(G, alpha=0.85)
    return pr, G

def build_index(corpus, strategy="advanced"):
    vec = build_vectorizer(corpus, strategy)
    X = vec.fit_transform(corpus.content.fillna("").tolist())
    toks, idf, avgdl = bm25_index(corpus.content.fillna("").tolist())
    pr, G = pagerank_scores(corpus)
    return {"vectorizer": vec, "X": X, "tokens": toks, "idf": idf, "avgdl": avgdl, "pagerank": pr, "graph": G}

def search_index(query, corpus, idx, top_k=10, alpha=0.80, phrase_boost=0.20):
    if corpus.empty or not query.strip():
        return pd.DataFrame()
    bm = bm25_scores(query, idx["tokens"], idx["idf"], idx["avgdl"])
    qv = idx["vectorizer"].transform([query])
    tfidf = cosine_similarity(qv, idx["X"]).ravel()
    pr = np.array([idx["pagerank"].get(i, 0.0) for i in corpus.doc_id])
    # Normalize each ranking component before fusion.
    bm_n, tf_n, pr_n = minmax(bm), minmax(tfidf), minmax(pr)
    scores = alpha * (0.65 * bm_n + 0.35 * tf_n) + (1-alpha) * pr_n

    qnorm = normalize_text(query)
    for i, txt in enumerate(corpus.content):
        if qnorm and qnorm in normalize_text(txt):
            scores[i] += phrase_boost

    out = corpus.copy()
    out["BM25"] = bm
    out["TFIDF"] = tfidf
    out["PageRank"] = pr
    out["Score"] = scores
    out = out.sort_values(["Score","BM25"], ascending=False).head(top_k).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out)+1)
    return out

# ----------------------------
# Metrics
# ----------------------------
def precision_recall_f1(retrieved, relevant):
    retrieved = set(retrieved)
    relevant = set(relevant)
    tp = len(retrieved & relevant)
    precision = tp / len(retrieved) if retrieved else 0.0
    recall = tp / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

def precision_at_k(retrieved, relevant, k):
    r = retrieved[:k]
    return len(set(r) & set(relevant)) / k if k else 0.0

def recall_at_k(retrieved, relevant, k):
    r = retrieved[:k]
    return len(set(r) & set(relevant)) / len(set(relevant)) if relevant else 0.0

def ap(retrieved, relevant):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)

def mrr(retrieved, relevant):
    relevant = set(relevant)
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1 / i
    return 0.0

def ndcg(retrieved, relevant):
    rel = set(relevant)
    gains = [1 if d in rel else 0 for d in retrieved]
    dcg = sum(g / np.log2(i+2) for i,g in enumerate(gains))
    ideal_n = min(len(rel), len(retrieved))
    idcg = sum(1 / np.log2(i+2) for i in range(ideal_n))
    return dcg / idcg if idcg else 0.0

def evaluate(retrieved, relevant, k=5):
    p,r,f = precision_recall_f1(retrieved, relevant)
    return {
        "Precision": p, "Recall": r, "F1": f,
        f"Precision@{k}": precision_at_k(retrieved, relevant, k),
        f"Recall@{k}": recall_at_k(retrieved, relevant, k),
        "MAP": ap(retrieved, relevant),
        "MRR": mrr(retrieved, relevant),
        "NDCG": ndcg(retrieved, relevant),
    }

# ----------------------------
# UI
# ----------------------------
st.title("🔎 AML Intelligence — End-to-End Information Retrieval System")
st.caption("BITS Pilani IR Assignment 2 | Crawling → Mining → Indexing → Search → PageRank → Recommendation → Evaluation")

corpus = load_corpus()

with st.sidebar:
    st.header("System Status")
    st.metric("Indexed documents", len(corpus))
    if not corpus.empty:
        st.metric("Unique domains", corpus.source_domain.nunique())
        st.metric("Avg. words/doc", int(corpus.word_count.mean()))
    st.divider()
    st.info("Recommended demo: crawl 20–50 pages from 2–3 seed domains, build the index, then use Search, Mining, Recommendations and Evaluation.")

tabs = st.tabs([
    "🏠 Dashboard","🕷️ Crawl","🗂️ Index","🔍 Search","📊 Ranking",
    "🧠 Text Mining","💡 Recommendations","📏 Evaluation","⚡ Performance"
])

# Dashboard
with tabs[0]:
    st.subheader("End-to-End Workflow")
    cols = st.columns(4)
    cols[0].metric("Documents", len(corpus))
    cols[1].metric("Domains", corpus.source_domain.nunique() if not corpus.empty else 0)
    cols[2].metric("Total words", f"{int(corpus.word_count.sum()):,}" if not corpus.empty else 0)
    cols[3].metric("Links", int(corpus.links_count.sum()) if not corpus.empty else 0)

    st.markdown("""
    **Use case:** Financial-crime intelligence retrieval. The system retrieves public AML/CFT advisories,
    regulatory guidance and financial-crime publications, ranks them with BM25 + TF-IDF + PageRank,
    and recommends similar documents.

    **IR lifecycle:** heterogeneous sources → configurable crawler → duplicate handling → separate
    content/metadata storage → preprocessing/feature engineering → inverted-style scoring → ranking →
    recommendations → IR evaluation → performance analytics.
    """)
    if not corpus.empty:
        st.dataframe(corpus[["doc_id","title","source_domain","depth","word_count","links_count"]].head(20),
                     use_container_width=True)

# Crawl
with tabs[1]:
    st.subheader("🕷️ Web Crawling Interface")
    seeds_text = st.text_area("Seed URLs (one per line)", value="\n".join(DEFAULT_SEEDS), height=120)
    c1,c2,c3 = st.columns(3)
    max_depth = c1.slider("Crawl depth", 0, 3, 1)
    max_pages = c2.slider("Max pages", 5, 100, 30)
    timeout = c3.slider("Timeout (seconds)", 5, 30, 12)
    stay = st.checkbox("Stay on the domains of the seed URLs", True)
    if st.button("🚀 Start Crawl", type="primary"):
        seeds = [x.strip() for x in seeds_text.splitlines() if x.strip()]
        if not seeds:
            st.error("Provide at least one seed URL.")
        else:
            with st.spinner("Crawling and extracting documents..."):
                pages, errors = crawl(seeds, max_depth, max_pages, stay, timeout)
                added, dup_url, dup_doc = persist_pages(pages)
            st.success(f"Crawl finished: {len(pages)} pages fetched, {added} new documents stored.")
            st.write(f"Duplicate URLs skipped: **{dup_url}** | Duplicate documents skipped: **{dup_doc}** | Errors/skips: **{len(errors)}**")
            if errors:
                with st.expander("View crawl issues"):
                    st.dataframe(pd.DataFrame(errors, columns=["URL","Issue"]), use_container_width=True)
            st.rerun()

# Index
with tabs[2]:
    st.subheader("🗂️ Index Management")
    corpus = load_corpus()
    if corpus.empty:
        st.warning("Crawl or upload data before building an index.")
    else:
        strategy = st.radio("Feature strategy", ["advanced","baseline"], horizontal=True,
                             help="Advanced = English stopword removal + unigrams/bigrams + sublinear TF. Baseline = raw unigrams.")
        if st.button("🏗️ Build / Refresh Index", type="primary"):
            with st.spinner("Building TF-IDF, BM25 statistics and PageRank graph..."):
                st.session_state["index"] = build_index(corpus, strategy)
                st.session_state["index_strategy"] = strategy
            st.success("Index built successfully.")
        idx = st.session_state.get("index")
        if idx:
            st.write(f"Strategy: **{st.session_state.get('index_strategy','advanced')}**")
            st.metric("Vocabulary size", len(idx["vectorizer"].vocabulary_))
            st.metric("Graph nodes", idx["graph"].number_of_nodes())
            st.metric("Graph edges", idx["graph"].number_of_edges())
            if st.button("🗑️ Clear all indexed corpus data"):
                con = db(); con.executescript("DELETE FROM links; DELETE FROM metadata; DELETE FROM documents;"); con.commit(); con.close()
                st.session_state.pop("index", None)
                st.success("Corpus cleared.")
                st.rerun()

# Search
with tabs[3]:
    st.subheader("🔍 Intelligent Search")
    corpus = load_corpus()
    if "index" not in st.session_state and not corpus.empty:
        st.session_state["index"] = build_index(corpus, "advanced")
        st.session_state["index_strategy"] = "advanced"
    idx = st.session_state.get("index")
    if not idx:
        st.info("Build the index first.")
    else:
        q = st.text_input("Enter a search query", placeholder="e.g., shell companies and digital assets AML red flags")
        c1,c2,c3 = st.columns(3)
        k = c1.slider("Top-K", 3, 20, 10)
        alpha = c2.slider("Text vs PageRank", 0.0, 1.0, 0.80, 0.05)
        phrase = c3.slider("Exact phrase boost", 0.0, 0.5, 0.20, 0.05)
        if q:
            t0=time.perf_counter()
            results = search_index(q, corpus, idx, k, alpha, phrase)
            elapsed=(time.perf_counter()-t0)*1000
            st.session_state["last_results"]=results
            st.session_state["last_query"]=q
            st.metric("Search latency", f"{elapsed:.1f} ms")
            for _, row in results.iterrows():
                with st.container(border=True):
                    st.markdown(f"**#{int(row.Rank)} — {row.title[:160]}**")
                    st.caption(f"{row.url} | {row.source_domain} | depth={row.depth} | {row.word_count:,} words")
                    st.write(row.content[:500] + ("..." if len(row.content)>500 else ""))
                    st.write(f"Final score **{row.Score:.4f}** · BM25 **{row.BM25:.4f}** · TF-IDF **{row.TFIDF:.4f}** · PageRank **{row.PageRank:.6f}**")

# Ranking
with tabs[4]:
    st.subheader("📊 Ranking Visualization")
    results = st.session_state.get("last_results")
    if results is None or results.empty:
        st.info("Run a search first.")
    else:
        plot_df = results[["Rank","title","BM25","TFIDF","PageRank","Score"]].copy()
        plot_df["label"] = plot_df["title"].str.slice(0,45)
        st.bar_chart(plot_df.set_index("label")[["BM25","TFIDF","Score"]])
        st.line_chart(plot_df.set_index("Rank")[["Score","PageRank"]])
        st.dataframe(results[["Rank","title","BM25","TFIDF","PageRank","Score","url"]], use_container_width=True)

# Text Mining
with tabs[5]:
    st.subheader("🧠 Text Preprocessing & Mining")
    corpus = load_corpus()
    if corpus.empty:
        st.info("Crawl data first.")
    else:
        a,b,c = st.columns(3)
        a.metric("Documents", len(corpus))
        b.metric("Total tokens", f"{int(corpus.word_count.sum()):,}")
        c.metric("Median words/doc", int(corpus.word_count.median()))
        st.markdown("### Corpus profile")
        st.bar_chart(corpus.groupby("source_domain").size().sort_values(ascending=False))
        st.markdown("### Document length distribution")
        hist = pd.DataFrame({"word_count": corpus.word_count})
        st.bar_chart(hist["word_count"].value_counts(bins=10).sort_index())

        doc_options = corpus.apply(lambda r: f'{r.doc_id} — {r.title[:100]}', axis=1).tolist()
        selected = st.selectbox("Document for keyword extraction", doc_options)
        did = int(selected.split(" — ")[0])
        row = corpus[corpus.doc_id==did].iloc[0]
        kws = keyword_extract(row.content, 15)
        st.write("**Top TF-IDF keywords:**", ", ".join([f"{w} ({s:.3f})" for w,s in kws]))

        st.markdown("### Unsupervised document classification / profiling")
        n_clusters = st.slider("Number of clusters", 2, min(8, len(corpus)), min(4, len(corpus)))
        vec = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1,2))
        X = vec.fit_transform(corpus.content)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        prof = corpus[["doc_id","title","source_domain"]].copy()
        prof["cluster"] = labels
        if len(set(labels)) > 1 and len(corpus) > n_clusters:
            sil = silhouette_score(X, labels)
            st.metric("Silhouette score", f"{sil:.3f}")
        st.dataframe(prof, use_container_width=True)

        st.markdown("### Feature strategy comparison")
        bvec = build_vectorizer(corpus, "baseline")
        avec = build_vectorizer(corpus, "advanced")
        BX=bvec.fit_transform(corpus.content)
        AX=avec.fit_transform(corpus.content)
        comp=pd.DataFrame([
            ["Baseline TF-IDF","Unigrams; no stopword removal",len(bvec.vocabulary_),BX.shape[1]],
            ["Advanced TF-IDF","English stopwords removed; 1-2 grams; sublinear TF",len(avec.vocabulary_),AX.shape[1]]
        ], columns=["Strategy","Preprocessing","Vocabulary","Matrix features"])
        st.dataframe(comp, use_container_width=True)

# Recommendations
with tabs[6]:
    st.subheader("💡 Content-Based Recommendation")
    corpus = load_corpus()
    if corpus.empty:
        st.info("Crawl data first.")
    else:
        vec = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1,2), sublinear_tf=True)
        X=vec.fit_transform(corpus.content)
        opts=corpus.apply(lambda r: f'{r.doc_id} — {r.title[:100]}', axis=1).tolist()
        sel=st.selectbox("Select a document", opts)
        k=st.slider("Top-K recommendations",3,15,5)
        did=int(sel.split(" — ")[0])
        i=corpus.index[corpus.doc_id==did][0]
        sims=cosine_similarity(X[i],X).ravel()
        order=np.argsort(sims)[::-1]
        rec=[]
        for j in order:
            if corpus.iloc[j].doc_id==did: continue
            rec.append([int(corpus.iloc[j].doc_id), corpus.iloc[j].title, corpus.iloc[j].source_domain, float(sims[j]), corpus.iloc[j].url])
            if len(rec)>=k: break
        st.dataframe(pd.DataFrame(rec,columns=["doc_id","title","domain","similarity","url"]), use_container_width=True)
        st.caption("Similarity is cosine similarity over TF-IDF unigram/bigram representations.")

# Evaluation
with tabs[7]:
    st.subheader("📏 IR Evaluation Dashboard")
    corpus = load_corpus()
    idx = st.session_state.get("index")
    if not idx:
        st.info("Build the index first.")
    else:
        st.caption("Assignment requirement: evaluate 3–5 queries using manual relevance judgements. Complete each query separately and then review the aggregate results.")
        default_queries = [
            "money laundering",
            "beneficial ownership",
            "suspicious activity reporting",
            "digital assets AML",
            "shell companies",
        ]
        if "eval_queries" not in st.session_state:
            last_q = st.session_state.get("last_query")
            st.session_state["eval_queries"] = [last_q] if last_q else default_queries[:3]
        if "eval_judgements" not in st.session_state:
            st.session_state["eval_judgements"] = {}

        nq = st.slider("Number of evaluation queries", 3, 5, min(5, max(3, len(st.session_state["eval_queries"]))))
        query_values = st.session_state["eval_queries"] + default_queries
        query_values = [q for q in query_values if q]
        while len(query_values) < nq:
            query_values.append(default_queries[len(query_values) % len(default_queries)])
        query_values = query_values[:nq]

        edited_queries = []
        for i in range(nq):
            edited_queries.append(st.text_input(f"Evaluation query {i+1}", value=query_values[i], key=f"eval_query_{i}"))
        st.session_state["eval_queries"] = edited_queries

        selected_q = st.selectbox("Query to judge", edited_queries, key="eval_selected_query")
        results = search_index(selected_q, corpus, idx, top_k=20)
        st.write("Select the documents that are relevant to the selected query (manual relevance judgement).")
        previous = set(st.session_state["eval_judgements"].get(selected_q, []))
        labels = []
        for _, r in results.iterrows():
            checked = st.checkbox(
                f"Relevant: #{int(r.Rank)} — {r.title[:130]}",
                value=int(r.doc_id) in previous,
                key=f"rel_{selected_q}_{int(r.doc_id)}"
            )
            labels.append(checked)
        relevant = [int(r.doc_id) for (_, r), lab in zip(results.iterrows(), labels) if lab]

        if st.button("Save Judgements & Calculate Metrics", type="primary"):
            st.session_state["eval_judgements"][selected_q] = relevant
            st.session_state["eval_metrics"] = st.session_state.get("eval_metrics", {})
            st.session_state["eval_metrics"][selected_q] = evaluate(results.doc_id.astype(int).tolist(), relevant, k=5)
            st.success(f"Saved {len(relevant)} relevant documents for: {selected_q}")

        saved = st.session_state.get("eval_metrics", {})
        if saved:
            st.markdown("### Per-query evaluation results")
            rows = []
            for qname, metrics in saved.items():
                rows.append({"Query": qname, **metrics})
            per_query = pd.DataFrame(rows)
            st.dataframe(per_query, use_container_width=True)

            numeric_cols = [c for c in per_query.columns if c != "Query"]
            st.markdown("### Aggregate evaluation (mean across judged queries)")
            aggregate = per_query[numeric_cols].mean().to_frame("Mean value")
            st.dataframe(aggregate, use_container_width=True)

            st.markdown("### Ranking-strategy comparison")
            comparison_rows = []
            for qname, rel in st.session_state["eval_judgements"].items():
                qresults = search_index(qname, corpus, idx, top_k=20)
                base = qresults.sort_values("BM25", ascending=False).doc_id.astype(int).tolist()
                final = qresults.sort_values("Score", ascending=False).doc_id.astype(int).tolist()
                comparison_rows.extend([
                    {"Query": qname, "Strategy": "BM25 only", **evaluate(base, rel, k=5)},
                    {"Query": qname, "Strategy": "BM25 + TF-IDF + PageRank", **evaluate(final, rel, k=5)},
                ])
            comp = pd.DataFrame(comparison_rows)
            st.dataframe(comp, use_container_width=True)
            if not comp.empty:
                summary = comp.groupby("Strategy")[["Precision@5", "Recall@5", "MAP", "MRR", "NDCG"]].mean()
                st.markdown("### Mean ranking-strategy performance")
                st.dataframe(summary, use_container_width=True)
                st.bar_chart(summary)

# Performance
with tabs[8]:
    st.subheader("⚡ Performance Analytics")
    corpus=load_corpus()
    if corpus.empty:
        st.info("No corpus available.")
    else:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Documents",len(corpus))
        c2.metric("Avg words",int(corpus.word_count.mean()))
        c3.metric("Max depth",int(corpus.depth.max()))
        c4.metric("Domains",corpus.source_domain.nunique())
        st.markdown("### Crawl depth profile")
        st.bar_chart(corpus.groupby("depth").size())
        st.markdown("### Source-domain distribution")
        st.bar_chart(corpus.groupby("source_domain").size().sort_values(ascending=False))
        if "index" in st.session_state:
            st.markdown("### PageRank top documents")
            pr=st.session_state["index"]["pagerank"]
            prdf=corpus[["doc_id","title","url"]].copy()
            prdf["PageRank"]=prdf.doc_id.map(pr).fillna(0)
            st.dataframe(prdf.sort_values("PageRank",ascending=False).head(15),use_container_width=True)

st.divider()
st.caption("Academic prototype for BITS Pilani Information Retrieval Assignment 2. Use only publicly accessible pages and keep crawl limits small.")
