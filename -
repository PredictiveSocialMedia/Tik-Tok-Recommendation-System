13:20:29 [INFO] train_full_pipeline: FULL TRAINING PIPELINE
13:20:29 [INFO] train_full_pipeline: DB: ...ler.supabase.com:5432/postgres
13:20:29 [INFO] train_full_pipeline: As-of: 2026-04-12T11:20:29.482781+00:00
13:20:29 [INFO] train_full_pipeline: Artifacts: /Users/arimoreira/Desktop/Tik-Tok-Recommendation-System/artifacts
13:20:29 [INFO] train_full_pipeline: [SKIPPED] Hashtag backfill
13:20:29 [INFO] train_full_pipeline: Step 2/7: Export canonical contract bundle from Supabase
13:20:41 [ERROR] train_full_pipeline: Export failed: cannot import name 'model_validator' from 'pydantic' (/Users/arimoreira/Desktop/Tik-Tok-Recommendation-System/.venv/lib/python3.11/site-packages/pydantic/__init__.cpython-311-darwin.so)
Traceback (most recent call last):
  File "/Users/arimoreira/Desktop/Tik-Tok-Recommendation-System/scripts/train_full_pipeline.py", line 543, in main
    results["export"] = step_export_bundle(
                        ^^^^^^^^^^^^^^^^^^^
  File "/Users/arimoreira/Desktop/Tik-Tok-Recommendation-System/scripts/train_full_pipeline.py", line 85, in step_export_bundle
    from scripts.export_db_contract_bundle import export_bundle_from_db
  File "/Users/arimoreira/Desktop/Tik-Tok-Recommendation-System/scripts/export_db_contract_bundle.py", line 20, in <module>
    from src.recommendation import CanonicalDatasetBundle, build_contract_manifest
  File "/Users/arimoreira/Desktop/Tik-Tok-Recommendation-System/src/recommendation/__init__.py", line 1, in <module>
    from .contracts import (
  File "/Users/arimoreira/Desktop/Tik-Tok-Recommendation-System/src/recommendation/contracts.py", line 10, in <module>
    from pydantic import BaseModel, Field, HttpUrl, ValidationError, model_validator
ImportError: cannot import name 'model_validator' from 'pydantic' (/Users/arimoreira/Desktop/Tik-Tok-Recommendation-System/.venv/lib/python3.11/site-packages/pydantic/__init__.cpython-311-darwin.so)
