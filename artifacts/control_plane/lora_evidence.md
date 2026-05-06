# LoRA hashtag fine-tuning evidence

- Status: **candidate_missing**
- Train Rows: **511**
- Validation Rows: **109**
- Test Rows: **111**
- Split: `test=/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/data/splits/test.jsonl, train=/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/data/splits/train.jsonl, validation=/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/data/splits/validation.jsonl`

| Model | Role | precision@10 | recall@10 | f1@10 |
|---|---|---|---|---|
| topic_prior_hashtag_baseline | baseline | 0.0789 | 0.1754 | 0.1057 |

- Primary comparison: **candidate_missing** on `f1@10`

## Notes
- LoRA adapter directory not found: /Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/models/tiktok-hashtag-lora. Run scripts/fine_tune_lora_hashtag.py first, then rerun this evidence report.
