# BITS Pilani IR Assignment 2 — Report Template

## 1. Title
**AML Intelligence: End-to-End Information Retrieval and Recommendation System**

## 2. Objective
State the assignment objective and explain why financial-crime intelligence was selected as the application domain.

## 3. Dataset / Sources
Document the exact seed URLs, crawl date, crawl depth, maximum pages, number of fetched pages, number of unique documents, duplicate URLs and duplicate documents.

## 4. System Architecture
Describe:
Crawling → Cleaning → Duplicate Handling → SQLite Storage → TF-IDF/BM25 Index → PageRank → Search/Ranking → Recommendation → Evaluation → Analytics.

## 5. Crawling
Include screenshots of the Crawl tab and results. Discuss multi-seed crawling, configurable depth, domain control and failure handling.

## 6. Preprocessing and Mining
Include:
- corpus size and document-length distribution
- keyword extraction example
- KMeans cluster/profile table
- baseline vs advanced feature representation comparison

## 7. Search and Ranking
Explain:
- BM25
- TF-IDF cosine similarity
- PageRank
- score fusion
- exact phrase boost

Show a query and Top-K results with component scores.

## 8. Recommendation
Explain content-based recommendation using TF-IDF cosine similarity. Show Top-K recommendations and similarity scores.

## 9. Evaluation
For 3–5 queries, record manual relevance judgements for each query and report both per-query and mean aggregate values:
Precision, Recall, F1, Precision@5, Recall@5, MAP, MRR, NDCG.

Compare BM25-only with the final fused ranking.

## 10. Experimental Results
Insert screenshots and tables from the Streamlit application.

## 11. Inference and Discussion
### Q1. Relevant documents but poor ranking
Discuss causes such as weak term weighting, query-document vocabulary mismatch, long-document bias, ambiguous queries and missing authority signals. Propose BM25 tuning, field weighting, query expansion, semantic retrieval and learning-to-rank.

### Q2. Duplicate / near-duplicate documents
Discuss their impact on indexing, ranking, recommendation and evaluation. Explain URL canonicalization, content hashing and near-duplicate detection.

### Q3. Content-based vs collaborative recommendation
Content-based is suitable for cold-start and item metadata-rich environments. Collaborative recommendation is preferable when there is a large history of user-item interactions and can capture latent preferences.

### Q4. End-to-end integration
Explain how crawling expands the collection, preprocessing standardizes content, indexing enables retrieval, ranking orders candidates, recommendation extends discovery, and evaluation closes the improvement loop.

### Q5. Learnings
Use the actual experimental observations. State which ranking strategy, preprocessing strategy and recommendation behavior worked best, and explain why.

## 12. Limitations and Future Work
Mention dynamic pages, robots restrictions, HTML-only extraction, manual relevance judgements, and possible future use of embeddings, semantic search, HITS, MinHash and learning-to-rank.

## 13. Conclusion
Summarize the end-to-end IR lifecycle and the measured results.
