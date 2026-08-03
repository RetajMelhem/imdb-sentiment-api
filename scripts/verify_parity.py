"""Compare production inference with an independent notebook-reference path."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sentiment.config import ModelMetadata, Settings, load_metadata
from sentiment.predictor import SentimentPredictor
from sentiment.preprocessing import encode_texts

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
GOLDEN_PATH = Path(__file__).with_name("notebook_parity.json")
GOLDEN_EXAMPLES = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["examples"]
PARITY_TEXTS = [
    *(example["text"] for example in GOLDEN_EXAMPLES),
    "Good &amp; strange.<br />The final scene worked.",
    "A carefully paced story with strong acting and vivid direction. " * 20,
    " ".join(f"token-{index}" for index in range(700)),
]


@dataclass(frozen=True)
class ParityReport:
    example_count: int
    max_logit_absolute_difference: float
    max_positive_probability_absolute_difference: float
    max_negative_probability_absolute_difference: float
    max_notebook_probability_absolute_difference: float
    validation_predictions_verified: bool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_exported_provenance(model_path: Path, metadata: ModelMetadata) -> None:
    model_files = list(model_path.glob("model*.safetensors"))
    if len(model_files) != 1:
        raise AssertionError("Export must contain exactly one safetensors model file.")
    expected_hashes = {
        model_path / "config.json": metadata.artifact_provenance.config_sha256,
        model_files[0]: metadata.artifact_provenance.model_sha256,
        model_path / "spm.model": metadata.artifact_provenance.tokenizer_spm_sha256,
    }
    for artifact_path, expected_hash in expected_hashes.items():
        if not artifact_path.is_file():
            raise AssertionError(
                f"Export is missing provenance artifact: {artifact_path}"
            )
        observed_hash = _file_sha256(artifact_path)
        if observed_hash != expected_hash:
            raise AssertionError(
                f"Artifact hash mismatch for {artifact_path.name}: "
                f"{observed_hash} != {expected_hash}"
            )


def reference_normalize(text: str) -> str:
    value = html.unescape(str(text))
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = CONTROL_CHARACTERS.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def reference_records(tokenizer, texts: list[str], max_length: int):
    records = []
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    for text in texts:
        content = tokenizer(
            reference_normalize(text),
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]
        if len(content) > budget:
            head_count = (budget + 1) // 2
            tail_count = budget - head_count
            content = content[:head_count] + (
                content[-tail_count:] if tail_count else []
            )
        prepare_for_model = getattr(tokenizer, "prepare_for_model", None)
        if callable(prepare_for_model):
            prepared = prepare_for_model(
                content,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=True,
            )
        else:
            cls_token_id = getattr(tokenizer, "cls_token_id", None)
            sep_token_id = getattr(tokenizer, "sep_token_id", None)
            if cls_token_id is None or sep_token_id is None:
                raise AssertionError("Tokenizer is missing CLS/SEP token IDs.")
            input_ids = [cls_token_id, *content, sep_token_id]
            prepared = {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }
        records.append(
            {
                "input_ids": prepared["input_ids"],
                "attention_mask": prepared["attention_mask"],
            }
        )
    return records


def reference_outputs(
    predictor: SentimentPredictor, texts: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    records = reference_records(predictor.tokenizer, texts, predictor.max_length)
    logits_rows = []
    with torch.inference_mode():
        for start in range(0, len(records), predictor.inference_batch_size):
            batch = predictor.tokenizer.pad(
                records[start : start + predictor.inference_batch_size],
                padding=True,
                pad_to_multiple_of=predictor.metadata.pad_to_multiple_of,
                return_tensors="pt",
            )
            batch = {
                key: value.to(predictor._torch_device) for key, value in batch.items()
            }
            logits = predictor.model(**batch).logits.float()
            logits_rows.append(logits.cpu())
    combined_logits = torch.cat(logits_rows)
    return combined_logits, torch.softmax(combined_logits, dim=-1)[:, 1]


def _maximum_absolute_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise AssertionError(f"Shape mismatch: {left.shape} != {right.shape}")
    return float(torch.max(torch.abs(left - right)).item())


def _binary_roc_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    ranks = np.empty(len(probability), dtype=np.float64)
    start = 0
    while start < len(probability):
        end = start + 1
        while (
            end < len(probability)
            and sorted_probability[end] == sorted_probability[start]
        ):
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive = labels == 1
    positive_count = int(positive.sum())
    negative_count = len(labels) - positive_count
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def verify_validation_predictions(path: Path, metadata: ModelMetadata) -> None:
    observed_hash = _file_sha256(path)
    expected_hash = metadata.artifact_provenance.validation_predictions_sha256
    if observed_hash != expected_hash:
        raise AssertionError(
            f"Validation prediction hash mismatch: {observed_hash} != {expected_hash}"
        )
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"positive_probability", "labels"}:
            raise AssertionError("Validation prediction artifact has unexpected keys.")
        probability = payload["positive_probability"].astype(np.float64, copy=True)
        labels = payload["labels"].astype(np.int64, copy=True)
    if probability.shape != (2481,) or labels.shape != probability.shape:
        raise AssertionError("Validation prediction artifact has unexpected shapes.")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise AssertionError("Validation probabilities must be finite and in [0, 1].")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise AssertionError("Validation labels must be binary.")

    predictions = (probability >= 0.5).astype(np.int64)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    accuracy = float(np.mean(predictions == labels))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probability, epsilon, 1 - epsilon)
    log_loss = float(
        -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    )
    observed = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "log_loss": log_loss,
        "roc_auc": _binary_roc_auc(labels, probability),
    }
    for name, value in observed.items():
        expected = metadata.final_seed_validation_metrics[name]
        if abs(value - expected) > 1e-12:
            raise AssertionError(
                f"Validation metric mismatch for {name}: {value} != {expected}"
            )


def verify(
    model_path: Path,
    atol: float,
    golden_atol: float = 1e-4,
    validation_predictions_path: Path | None = None,
) -> ParityReport:
    metadata_path = model_path.resolve() / "model_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Exported model is missing its colocated model_metadata.json artifact."
        )
    metadata = load_metadata(metadata_path)
    verify_exported_provenance(model_path.resolve(), metadata)
    settings = Settings(
        model_path=model_path,
        model_id=None,
        metadata_path=metadata_path,
        max_length=None,
        decision_threshold=None,
        inference_batch_size=None,
    )
    predictor = SentimentPredictor(settings, metadata)
    reference_tokens = reference_records(
        predictor.tokenizer, PARITY_TEXTS, metadata.max_length
    )
    production_tokens = encode_texts(
        PARITY_TEXTS, predictor.tokenizer, metadata.max_length
    )
    if production_tokens != reference_tokens:
        raise AssertionError("Production tokenization differs from the notebook path.")
    content_lengths = [
        len(
            predictor.tokenizer(
                reference_normalize(text),
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]
        )
        for text in PARITY_TEXTS
    ]
    content_budget = metadata.max_length - 2
    if not 100 < content_lengths[-2] <= content_budget:
        raise AssertionError("Long in-budget parity case is not in the expected range.")
    if content_lengths[-1] <= content_budget:
        raise AssertionError("Overflow parity case does not exceed max_length.")
    if len(production_tokens[-1]["input_ids"]) != metadata.max_length:
        raise AssertionError(
            "Overflow parity case did not use the full head-tail budget."
        )
    for index, example in enumerate(GOLDEN_EXAMPLES):
        actual_ids = production_tokens[index]["input_ids"]
        if actual_ids != example["input_ids"]:
            raise AssertionError(
                f"Tokenizer checkpoint mismatch at notebook example {index}."
            )
    reference_logits, reference_probability = reference_outputs(predictor, PARITY_TEXTS)
    captured_logits: list[torch.Tensor] = []

    def capture_logits(_module, _inputs, output) -> None:
        captured_logits.append(output.logits.detach().float().cpu())

    hook = predictor.model.register_forward_hook(capture_logits)
    try:
        production = predictor.predict_batch(PARITY_TEXTS)
    finally:
        hook.remove()
    production_logits = torch.cat(captured_logits)
    production_positive = torch.tensor(
        [item.positive_probability for item in production], dtype=torch.float32
    )
    production_negative = torch.tensor(
        [item.negative_probability for item in production], dtype=torch.float32
    )
    reference_negative = 1.0 - reference_probability
    max_logit_difference = _maximum_absolute_difference(
        reference_logits, production_logits
    )
    max_positive_difference = _maximum_absolute_difference(
        reference_probability, production_positive
    )
    max_negative_difference = _maximum_absolute_difference(
        reference_negative, production_negative
    )
    if max_logit_difference > atol:
        raise AssertionError(
            f"Maximum logit difference {max_logit_difference} exceeds {atol}."
        )
    if max_positive_difference > atol or max_negative_difference > atol:
        raise AssertionError("Production probabilities differ from notebook logic.")

    for index, (expected, actual) in enumerate(
        zip(reference_probability.tolist(), production, strict=True)
    ):
        if abs(expected - actual.positive_probability) > atol:
            raise AssertionError(
                f"Probability mismatch at example {index}: {expected} != "
                f"{actual.positive_probability}"
            )
        expected_label = (
            "positive" if expected >= metadata.decision_threshold else "negative"
        )
        expected_label_id = int(expected >= metadata.decision_threshold)
        if actual.label != expected_label or actual.label_id != expected_label_id:
            raise AssertionError(f"Label mismatch at example {index}")
        if abs(actual.confidence - max(expected, 1 - expected)) > atol:
            raise AssertionError(f"Confidence mismatch at example {index}")
    notebook_differences = []
    for index, example in enumerate(GOLDEN_EXAMPLES):
        expected = example["positive_probability"]
        actual = production[index].positive_probability
        notebook_differences.append(abs(expected - actual))
        if abs(expected - actual) > golden_atol:
            raise AssertionError(
                f"Checkpoint mismatch at notebook example {index}: "
                f"recorded={expected}, actual={actual}"
            )
    validation_verified = False
    if validation_predictions_path is not None:
        verify_validation_predictions(validation_predictions_path, metadata)
        validation_verified = True
    report = ParityReport(
        example_count=len(PARITY_TEXTS),
        max_logit_absolute_difference=max_logit_difference,
        max_positive_probability_absolute_difference=max_positive_difference,
        max_negative_probability_absolute_difference=max_negative_difference,
        max_notebook_probability_absolute_difference=max(notebook_differences),
        validation_predictions_verified=validation_verified,
    )
    print(f"Parity verified for {report.example_count} examples.")
    print(f"max_logit_abs_diff={report.max_logit_absolute_difference:.12g}")
    print(
        "max_positive_probability_abs_diff="
        f"{report.max_positive_probability_absolute_difference:.12g}"
    )
    print(
        "max_negative_probability_abs_diff="
        f"{report.max_negative_probability_absolute_difference:.12g}"
    )
    print(
        "max_notebook_rounded_probability_abs_diff="
        f"{report.max_notebook_probability_absolute_difference:.12g}"
    )
    print(f"validation_predictions_verified={validation_verified}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--golden-atol", type=float, default=1e-4)
    parser.add_argument("--validation-predictions", type=Path)
    args = parser.parse_args()
    verify(
        args.model_path,
        args.atol,
        args.golden_atol,
        args.validation_predictions,
    )


if __name__ == "__main__":
    main()
