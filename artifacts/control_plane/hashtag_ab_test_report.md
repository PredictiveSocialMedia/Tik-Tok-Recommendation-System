# Hashtag recommender A/B test

- Evaluable rows: **109** of 111
- Unique ground-truth hashtags: **409**

| Variant | Precision@5 | Recall@5 | F1@5 | Diversity@5 | Precision@10 | Recall@10 | F1@10 | Diversity@10 | Catalog Coverage |
|---|---|---|---|---|---|---|---|---|---|
| tfidf_baseline | 0.323 | 0.345 | 0.324 | 0.943 | 0.208 | 0.439 | 0.271 | 0.949 | 0.824 |
| hashtag_recommender | 0.334 | 0.372 | 0.340 | 0.811 | 0.250 | 0.518 | 0.321 | 0.789 | 1.249 |
