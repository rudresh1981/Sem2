# AML Intelligence — End-to-End IR System

A Streamlit implementation for the BITS Pilani Information Retrieval Assignment 2.

## Use case
Financial-crime intelligence retrieval: search public AML/CFT advisories, regulatory guidance and financial-crime publications.

## Assignment coverage
- Streamlit dashboard
- Configurable multi-seed web crawling and crawl depth
- Duplicate URL and duplicate-document handling
- Separate SQLite storage for document content and metadata
- TF-IDF feature engineering and keyword extraction
- Document profiling and KMeans-based unsupervised classification
- Baseline vs advanced preprocessing comparison
- BM25 retrieval
- TF-IDF cosine similarity
- PageRank link analysis
- Fused ranking with PageRank
- Ranking visualizations
- Content-based Top-K recommendation with cosine similarity
- Precision, Recall, F1, P@K, R@K, MAP, MRR and NDCG across 3–5 manually judged queries
- BM25-only vs fused ranking comparison
- Performance analytics

## Install
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Recommended demo flow
1. Open **Crawl**.
2. Keep the default FinCEN/FATF seeds or add other public sources.
3. Start with depth=1 and 20–40 pages.
4. Open **Index** and build the advanced index.
5. Use **Search** for queries such as:
   - shell companies and money laundering
   - digital assets AML red flags
   - suspicious activity reporting
   - beneficial ownership
   - fraud and financial crime
6. Open **Ranking** to show BM25, TF-IDF, PageRank and fused score.
7. Open **Text Mining** to show corpus profile, keywords, KMeans clusters and feature-strategy comparison.
8. Open **Recommendations** and show Top-K similar documents.
9. Open **Evaluation**, select 3–5 queries, manually judge relevant search results for each query, save the judgements, then review per-query and aggregate metrics and compare BM25-only with fused ranking.
10. Capture screenshots from the running Streamlit UI for the report and demo evidence.

## Data storage
The app creates `ir_system.db` automatically. It stores:
- `documents`: URL, title, content and content hash.
- `metadata`: source domain, seed URL, crawl depth, timestamp, word/character counts, content type, language and link count.
- `links`: source document to target URL edges for PageRank.

## Notes for the BITS lab
Use small crawl limits. Some websites may deny automated requests, require JavaScript, or expose PDFs rather than HTML. The app reports skipped URLs and errors rather than failing the complete crawl.

## Report discussion points
1. Poor ranking despite relevant retrieval can result from weak term weighting, vocabulary mismatch, long-document bias, query ambiguity, or insufficient link signals. Improvements include BM25 tuning, query expansion, field weighting, semantic embeddings and learning-to-rank.
2. Duplicate/near-duplicate pages inflate term statistics, distort PageRank, crowd recommendation results and can artificially improve evaluation. Mitigate using canonical URLs, content hashes and near-duplicate similarity/MinHash.
3. Content-based recommendation works well for cold-start and item-rich settings; collaborative recommendation works better when many user-item interactions exist and can discover non-obvious preferences.
4. The end-to-end pipeline makes each stage feed the next: crawling expands the corpus, preprocessing makes text comparable, indexing enables efficient retrieval, ranking improves ordering, recommendations extend discovery, and evaluation quantifies effectiveness.
5. Base final learnings on your actual screenshots and metric values rather than invented numbers.
