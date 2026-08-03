"""Reusable, batched Transformer sentiment predictor."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any

import torch

from sentiment.config import ModelMetadata, Settings
from sentiment.exceptions import InferenceBusyError, ModelLoadError, PredictionError
from sentiment.preprocessing import encode_texts
from sentiment.schemas import PredictionResult

LOGGER = logging.getLogger(__name__)


class SentimentPredictor:
    """Load model artifacts once and serve single or batched predictions."""

    def __init__(
        self,
        settings: Settings,
        metadata: ModelMetadata,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.settings = settings
        self.metadata = metadata
        self.max_length = settings.effective_max_length(metadata)
        self.decision_threshold = settings.effective_threshold(metadata)
        self.inference_batch_size = settings.effective_inference_batch_size(metadata)
        self._inference_gate = threading.BoundedSemaphore(
            settings.max_concurrent_inferences
        )
        self._torch_device = self._resolve_device(settings.device)
        self.device = str(self._torch_device)
        if tokenizer is None or model is None:
            tokenizer, model = self._load_components()
        self.tokenizer = tokenizer
        self.model = model
        self._validate_artifacts()
        self.model.to(self._torch_device)
        self.model.eval()
        LOGGER.info("Model initialized on device=%s", self.device)

    @staticmethod
    def _resolve_device(configured: str) -> torch.device:
        if configured == "cuda":
            if not torch.cuda.is_available():
                raise ModelLoadError(
                    "DEVICE=cuda was requested but CUDA is unavailable."
                )
            return torch.device("cuda")
        if configured == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def _load_components(self) -> tuple[Any, Any]:
        source = self.settings.model_source()
        source_type = "local" if self.settings.model_path is not None else "hub"
        LOGGER.info("Loading fine-tuned model from source_type=%s", source_type)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            common: dict[str, Any] = {
                "trust_remote_code": False,
            }
            if self.settings.hf_revision:
                common["revision"] = self.settings.hf_revision
            if self.settings.hf_home:
                common["cache_dir"] = str(self.settings.hf_home)
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                use_fast=self.metadata.tokenizer.use_fast,
                **common,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                source,
                use_safetensors=True,
                **common,
            )
            return tokenizer, model
        except Exception as error:
            LOGGER.exception("Failed to load configured model artifacts")
            raise ModelLoadError(
                "Unable to load the fine-tuned model artifacts."
            ) from error

    def _validate_artifacts(self) -> None:
        # Transformers 5 may expose the Rust-backed DeBERTa tokenizer even when
        # ``use_fast=False`` is requested. Exact token-ID parity is enforced by
        # scripts/verify_parity.py instead of relying on this mutable flag.
        if self.tokenizer.num_special_tokens_to_add(pair=False) != 2:
            raise ModelLoadError("Tokenizer must add exactly two special tokens.")
        config = getattr(self.model, "config", None)
        if config is None or int(getattr(config, "num_labels", -1)) != 2:
            raise ModelLoadError("Model must expose exactly two classification labels.")
        actual_id2label = {
            int(key): str(value).lower()
            for key, value in dict(getattr(config, "id2label", {})).items()
        }
        if actual_id2label != self.metadata.id2label:
            raise ModelLoadError("Model label mapping does not match audited metadata.")

    def predict(self, text: str) -> PredictionResult:
        """Predict one review using the shared batch implementation."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[PredictionResult]:
        """Predict a review batch with chunked, true batched model calls."""
        if not texts or not all(
            isinstance(text, str) and text.strip() for text in texts
        ):
            raise ValueError("Provide one or more non-empty review strings.")
        acquired = self._inference_gate.acquire(
            timeout=self.settings.inference_queue_timeout_seconds
        )
        if not acquired:
            raise InferenceBusyError("Inference capacity is temporarily busy.")
        try:
            return self._predict_batch(texts)
        finally:
            self._inference_gate.release()

    def _predict_batch(self, texts: Sequence[str]) -> list[PredictionResult]:
        probabilities: list[float] = []
        try:
            with torch.inference_mode():
                for start in range(0, len(texts), self.inference_batch_size):
                    text_chunk = texts[start : start + self.inference_batch_size]
                    chunk = encode_texts(text_chunk, self.tokenizer, self.max_length)
                    batch = self.tokenizer.pad(
                        chunk,
                        padding=True,
                        pad_to_multiple_of=self.metadata.pad_to_multiple_of,
                        return_tensors="pt",
                    )
                    device_batch = {
                        key: value.to(self._torch_device)
                        for key, value in batch.items()
                    }
                    logits = self.model(**device_batch).logits.float()
                    if logits.ndim != 2 or logits.shape[1] != 2:
                        raise ValueError("Model returned an invalid logits shape.")
                    chunk_probability = torch.softmax(logits, dim=-1)[:, 1]
                    if not torch.isfinite(chunk_probability).all():
                        raise ValueError("Model returned non-finite probabilities.")
                    probabilities.extend(chunk_probability.cpu().tolist())
        except Exception as error:
            LOGGER.exception("Inference failed for batch_size=%d", len(texts))
            raise PredictionError("Inference failed.") from error
        return [self._to_result(probability) for probability in probabilities]

    def _to_result(self, positive_probability: float) -> PredictionResult:
        label_id = int(positive_probability >= self.decision_threshold)
        negative_probability = 1.0 - positive_probability
        return PredictionResult(
            label=self.metadata.id2label[label_id],
            label_id=label_id,
            confidence=max(positive_probability, negative_probability),
            positive_probability=positive_probability,
            negative_probability=negative_probability,
        )
