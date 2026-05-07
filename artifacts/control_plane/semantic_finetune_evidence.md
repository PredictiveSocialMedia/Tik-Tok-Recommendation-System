# Semantic fine-tuning evidence

- Status: **candidate_missing**
- Train Rows: **511**
- Validation Rows: **109**
- Test Rows: **111**
- Split: `test=/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/data/splits/test.jsonl, train=/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/data/splits/train.jsonl, validation=/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/data/splits/validation.jsonl`

| Model | Role | ndcg@10 | mrr@10 |
|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | baseline_sbert | 1.0000 | 1.0000 |

- Primary comparison: **candidate_missing** on `ndcg@10`

## Notes
- Fine-tuned model directory not found: /Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/models/tiktok-sbert. Run scripts/fine_tune_embeddings.py first, then rerun this evidence report.
