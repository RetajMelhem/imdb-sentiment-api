"""Pydantic contracts shared by the predictor and HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_REVIEW_CHARS = 100_000
MAX_SCHEMA_BATCH_SIZE = 256


def _validate_review(value: str) -> str:
    if not value.strip():
        raise ValueError("Review text must not be blank.")
    return value


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionRequest(StrictRequest):
    text: str = Field(min_length=1, max_length=MAX_REVIEW_CHARS)

    _non_blank = field_validator("text")(_validate_review)


class BatchPredictionRequest(StrictRequest):
    texts: list[str] = Field(min_length=1, max_length=MAX_SCHEMA_BATCH_SIZE)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("At least one review is required.")
        for value in values:
            _validate_review(value)
        return values


class PredictionResult(BaseModel, frozen=True):
    label: Literal["negative", "positive"]
    label_id: int
    confidence: float
    positive_probability: float
    negative_probability: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    task: str
    model: str
    model_source: str
    dataset: str
    max_length: int
    threshold: float
    labels: list[str]
    preprocessing: str
    truncation_strategy: str
    test_metrics: dict[str, Any]
    frozen_config_fingerprint: str


class RootResponse(BaseModel):
    name: str
    version: str
    docs: str
    health: str
    model_info: str
