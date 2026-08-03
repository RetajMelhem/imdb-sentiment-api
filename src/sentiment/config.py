"""Central deployment settings and audited model metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sentiment.exceptions import ModelConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "model_metadata.json"


class TokenizerMetadata(BaseModel, frozen=True):
    source: str
    use_fast: bool


class ModelSelectionMetadata(BaseModel, frozen=True):
    candidates_screened: int
    selected_run: str
    seed: int
    ensemble_weights: tuple[float, ...]
    use_tta: bool


class ArtifactProvenance(BaseModel, frozen=True):
    run_fingerprint: str
    config_sha256: str
    model_sha256: str
    tokenizer_spm_sha256: str
    validation_predictions_sha256: str
    training_runtime: dict[str, str]


class ModelMetadata(BaseModel, frozen=True):
    """Validated single source of truth extracted from the notebook."""

    schema_version: int
    task: str
    dataset: str
    base_model: str
    model_selection: ModelSelectionMetadata
    train_config: dict[str, object]
    tokenizer: TokenizerMetadata
    num_labels: int
    id2label: dict[int, str]
    label2id: dict[str, int]
    positive_class_id: int
    preprocessing: str
    truncation_strategy: str
    max_length: int
    padding: str
    pad_to_multiple_of: int
    decision_threshold: float
    inference_batch_size: int
    probability_function: str
    validation_metrics: dict[str, float]
    final_seed_validation_metrics: dict[str, float]
    test_metrics: dict[str, object]
    frozen_config_fingerprint: str
    artifact_provenance: ArtifactProvenance

    @field_validator("decision_threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("decision_threshold must be in [0, 1]")
        return value


class Settings(BaseSettings):
    """Environment-driven deployment settings.

    The audited defaults preserve notebook behavior. Changing ``MAX_LENGTH`` or
    ``DECISION_THRESHOLD`` is an explicit behavior override and is surfaced by
    ``/model-info``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    model_path: Path | None = None
    model_id: str | None = None
    metadata_path: Path | None = None
    max_length: int | None = Field(default=None, gt=2, le=4096)
    decision_threshold: float | None = Field(default=None, ge=0, le=1)
    inference_batch_size: int | None = Field(default=None, ge=1, le=256)
    max_batch_size: int = Field(default=32, ge=1, le=1024)
    max_text_chars: int = Field(default=100_000, ge=1000, le=100_000)
    max_request_bytes: int = Field(default=4_194_304, ge=1024, le=67_108_864)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    hf_home: Path | None = None
    hf_revision: str | None = None
    max_concurrent_inferences: int = Field(default=1, ge=1, le=32)
    inference_queue_timeout_seconds: float = Field(default=5.0, gt=0, le=300)

    @field_validator("model_id")
    @classmethod
    def normalize_model_id(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None

    def model_source(self) -> str:
        """Resolve a fine-tuned source, with local artifacts taking precedence."""
        if self.model_path is not None:
            resolved = self.model_path.expanduser().resolve()
            if not resolved.is_dir():
                raise ModelConfigurationError(
                    f"Configured MODEL_PATH is not a directory: {resolved}"
                )
            return str(resolved)
        if self.model_id:
            if self.model_id == "microsoft/deberta-v3-base":
                raise ModelConfigurationError(
                    "MODEL_ID points to the unfine-tuned base checkpoint. Configure "
                    "the exported fine-tuned repository instead."
                )
            if not self.hf_revision or not re.fullmatch(
                r"[0-9a-fA-F]{40}", self.hf_revision
            ):
                raise ModelConfigurationError(
                    "Hub models require HF_REVISION set to an immutable 40-character "
                    "commit SHA."
                )
            return self.model_id
        raise ModelConfigurationError(
            "Configure MODEL_PATH or MODEL_ID for the exported fine-tuned model. "
            "The public base checkpoint is intentionally not used as a fallback."
        )

    def resolved_metadata_path(self) -> Path:
        """Prefer explicit or model-colocated metadata, then the audited repo copy."""
        if self.metadata_path is not None:
            return self.metadata_path.expanduser().resolve()
        if self.model_path is not None:
            colocated = self.model_path.expanduser().resolve() / "model_metadata.json"
            if colocated.is_file():
                return colocated
        working_copy = Path.cwd().resolve() / "model_metadata.json"
        if working_copy.is_file():
            return working_copy
        return DEFAULT_METADATA_PATH

    def effective_max_length(self, metadata: ModelMetadata) -> int:
        return self.max_length if self.max_length is not None else metadata.max_length

    def effective_threshold(self, metadata: ModelMetadata) -> float:
        return (
            self.decision_threshold
            if self.decision_threshold is not None
            else metadata.decision_threshold
        )

    def effective_inference_batch_size(self, metadata: ModelMetadata) -> int:
        return (
            self.inference_batch_size
            if self.inference_batch_size is not None
            else metadata.inference_batch_size
        )


def load_metadata(path: str | Path = DEFAULT_METADATA_PATH) -> ModelMetadata:
    """Load and validate the notebook-audited metadata artifact."""
    metadata_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelConfigurationError(
            f"Unable to load model metadata: {metadata_path}"
        ) from error
    return ModelMetadata.model_validate(payload)


def load_configured_metadata(settings: Settings) -> ModelMetadata:
    """Load metadata from the same immutable source selected for model weights."""
    if settings.metadata_path is not None or settings.model_path is not None:
        return load_metadata(settings.resolved_metadata_path())
    if settings.model_id is not None:
        settings.model_source()
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=settings.model_id,
                filename="model_metadata.json",
                revision=settings.hf_revision,
                cache_dir=str(settings.hf_home) if settings.hf_home else None,
            )
        except Exception as error:
            raise ModelConfigurationError(
                "Unable to load model_metadata.json from the pinned Hub revision."
            ) from error
        return load_metadata(downloaded)
    return load_metadata(settings.resolved_metadata_path())
