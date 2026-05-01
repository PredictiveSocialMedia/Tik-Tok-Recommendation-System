"""LoRA fine-tuning of a small causal LLM for TikTok hashtag generation.

Given a video caption + visual topics, the model learns to generate
contextually relevant, TikTok-native hashtags — replacing the rule-based
suggest_hashtags() in caption_suggest.py.

Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0  (1.1B params, ~2.2 GB)
Adapter:    LoRA r=8 applied to q_proj + v_proj via PEFT
Trainer:    trl SFTTrainer, causal LM on full prompt+completion text

Hard deps (training): peft>=0.10, trl>=0.8, transformers>=4.40, torch>=2.1
Optional (memory):    bitsandbytes>=0.43  (enable with load_in_4bit=True)

The pure-Python helpers (_extract_hashtags, _eligible, _build_prompt,
_build_target, parse_hashtags, build_dataset, evaluate) run without any
ML dependencies so they can be exercised in unit tests via a mock model.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from trl import SFTConfig, SFTTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _LORA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LORA_AVAILABLE = False

try:
    from datasets import Dataset as HFDataset  # type: ignore

    _DATASETS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DATASETS_AVAILABLE = False


# Prompt template: kept short so it fits within max_seq_len together with tags
_PROMPT_TEMPLATE = """\
### Task: Generate TikTok hashtags for this video.

### Caption:
{caption}

### Topics:
{topics}

### Hashtags:
"""


@dataclass
class LoraHashtagFinetunerConfig:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "models/tiktok-hashtag-lora"
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    # Training
    epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 4
    max_seq_len: int = 256
    learning_rate: float = 2e-4
    fp16: bool = False
    load_in_4bit: bool = False
    # Data filtering
    min_hashtags: int = 2
    max_hashtags_target: int = 10
    # Evaluation
    eval_k: int = 10
    seed: int = 42


class LoraHashtagFinetuner:
    """Fine-tune a small causal LLM with LoRA for TikTok hashtag generation.

    Training input  (prompt):  Caption + visual topics
    Training output (target):  Space-separated #hashtag tokens
    """

    def __init__(self, cfg: Optional[LoraHashtagFinetunerConfig] = None) -> None:
        self.cfg = cfg or LoraHashtagFinetunerConfig()

    # ------------------------------------------------------------------
    # Pure-Python helpers — no ML dependency
    # ------------------------------------------------------------------

    def _extract_hashtags(self, row: Dict[str, Any]) -> List[str]:
        """Return normalized hashtag strings (no leading #) from a row."""
        tags: List[str] = []
        seen: set = set()

        def _add(raw: str) -> None:
            clean = raw.strip().lower().lstrip("#").strip()
            if clean and len(clean) > 1 and clean not in seen:
                seen.add(clean)
                tags.append(clean)

        for tag in row.get("hashtags") or []:
            _add(str(tag))
        for word in str(row.get("caption") or "").split():
            if word.startswith("#"):
                _add(word)
        return tags

    def _eligible(self, row: Dict[str, Any]) -> bool:
        return len(self._extract_hashtags(row)) >= self.cfg.min_hashtags

    def _build_prompt(self, row: Dict[str, Any]) -> str:
        caption = str(row.get("caption") or row.get("text") or "").strip()
        topic_parts = list(row.get("keywords") or row.get("visual_topics") or [])
        topics = ", ".join(str(t) for t in topic_parts) or "none"
        return _PROMPT_TEMPLATE.format(caption=caption, topics=topics)

    def _build_target(self, row: Dict[str, Any]) -> str:
        tags = self._extract_hashtags(row)[: self.cfg.max_hashtags_target]
        return " ".join(f"#{t}" for t in tags)

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def build_dataset(self, rows: List[Dict[str, Any]]) -> "HFDataset":
        """Build an HF Dataset with a single 'text' column (prompt + target)."""
        if not _DATASETS_AVAILABLE:
            raise ImportError("datasets package required — pip install datasets")
        eligible = [r for r in rows if self._eligible(r)]
        texts = [self._build_prompt(r) + self._build_target(r) for r in eligible]
        logger.info(
            "Dataset: %d eligible rows (of %d total, min_hashtags=%d)",
            len(eligible),
            len(rows),
            self.cfg.min_hashtags,
        )
        return HFDataset.from_dict({"text": texts})

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_rows: List[Dict[str, Any]],
        val_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Any, Any]:
        """Fine-tune base model with LoRA and save adapter weights.

        Returns (peft_model, tokenizer).
        """
        if not _LORA_AVAILABLE:
            raise ImportError(
                "peft, trl, and transformers are required. "
                "Install with: pip install 'peft>=0.10' 'trl>=0.8' 'transformers>=4.40'"
            )
        if not _DATASETS_AVAILABLE:
            raise ImportError("datasets package required — pip install datasets")

        train_dataset = self.build_dataset(train_rows)
        if len(train_dataset) == 0:
            raise ValueError(
                "No eligible training rows — lower min_hashtags or provide more data."
            )
        val_dataset = self.build_dataset(val_rows) if val_rows else None

        output = Path(self.cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        bnb_config = None
        if self.cfg.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("4-bit quantization enabled via bitsandbytes")
            except ImportError:
                logger.warning(
                    "bitsandbytes not available — loading model in full precision"
                )

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.base_model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )

        lora_cfg = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        sft_cfg = SFTConfig(
            output_dir=str(output),
            num_train_epochs=self.cfg.epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_accum_steps,
            learning_rate=self.cfg.learning_rate,
            max_seq_length=self.cfg.max_seq_len,
            fp16=self.cfg.fp16,
            seed=self.cfg.seed,
            dataset_text_field="text",
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )
        trainer.train()

        model.save_pretrained(str(output))
        tokenizer.save_pretrained(str(output))
        logger.info("LoRA adapter and tokenizer saved to %s", output)
        return model, tokenizer

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @staticmethod
    def parse_hashtags(text: str) -> List[str]:
        """Extract normalized hashtag strings (no #) from raw generated text."""
        tags: List[str] = []
        seen: set = set()
        for token in text.split():
            m = re.match(r"^#([a-zA-Z0-9_]+)", token)
            if m:
                tag = m.group(1).lower()
                if tag not in seen and len(tag) > 1:
                    seen.add(tag)
                    tags.append(tag)
        return tags

    def _model_generate(self, model: Any, tokenizer: Any, prompt: str) -> str:
        """Run one forward pass and return the raw decoded completion.

        Extracted into its own method so tests can monkeypatch it without
        needing torch or a real model.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch>=2.1 is required for generation") from exc

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_seq_len,
        )
        device = "cpu"
        if hasattr(model, "parameters"):
            try:
                device = next(model.parameters()).device
            except StopIteration:
                pass
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        row: Dict[str, Any],
        k: Optional[int] = None,
    ) -> List[str]:
        """Generate hashtags for one row. Returns list of tag strings (no #)."""
        k = k or self.cfg.eval_k
        prompt = self._build_prompt(row)
        raw = self._model_generate(model, tokenizer, prompt)
        return self.parse_hashtags(raw)[:k]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        test_rows: List[Dict[str, Any]],
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute precision@k, recall@k, F1@k against ground-truth hashtags.

        predicted = generate(row)[:k]
        relevant  = all ground-truth hashtags in the row
        """
        k = k or self.cfg.eval_k
        zero = {f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"f1@{k}": 0.0}
        eligible = [r for r in test_rows if self._eligible(r)]
        if not eligible:
            return zero

        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []

        for row in eligible:
            predicted = set(self.generate(model, tokenizer, row, k=k))
            actual = set(self._extract_hashtags(row))
            tp = len(predicted & actual)
            p = tp / max(1, len(predicted))
            r = tp / max(1, len(actual))
            f = 2 * p * r / max(1e-9, p + r)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        return {
            f"precision@{k}": float(np.mean(precisions)),
            f"recall@{k}": float(np.mean(recalls)),
            f"f1@{k}": float(np.mean(f1s)),
        }
