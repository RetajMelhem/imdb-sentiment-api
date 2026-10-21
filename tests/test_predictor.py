from __future__ import annotations

import math

import pytest
import torch
from conftest import FakeModel

from sentiment.config import Settings, load_metadata
from sentiment.exceptions import PredictionError
from sentiment.predictor import SentimentPredictor


def make_predictor(metadata_path, fake_tokenizer, model, batch_size: int = 8):
    settings = Settings(
        _env_file=None,
        model_path=None,
        model_id="owner/fine-tuned",
        device="cpu",
        inference_batch_size=batch_size,
    )
    return SentimentPredictor(
        settings=settings,
        metadata=load_metadata(metadata_path),
        tokenizer=fake_tokenizer,
        model=model,
    )


def logit_for_probability(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def test_initialization_moves_and_evaluates_model_once(
    metadata_path, fake_tokenizer, fake_model
) -> None:
    predictor = make_predictor(metadata_path, fake_tokenizer, fake_model)
    predictor.predict("good movie")
    predictor.predict("another good movie")

    assert fake_model.device_moves == ["cpu"]
    assert fake_model.eval_calls == 1
    assert fake_model.forward_calls == 2


@pytest.mark.parametrize("available, expected", [(True, "cuda"), (False, "cpu")])
def test_auto_prioritizes_available_gpu(
    metadata_path, fake_tokenizer, fake_model, monkeypatch, available, expected
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    predictor = SentimentPredictor(
        Settings(_env_file=None, device="auto"),
        load_metadata(metadata_path),
        tokenizer=fake_tokenizer,
        model=fake_model,
    )
    assert predictor.device == expected
    assert fake_model.device_moves == [expected]


def test_auto_recovers_on_cpu_if_gpu_cannot_load_model(
    metadata_path, fake_tokenizer, monkeypatch
) -> None:
    class BusyGPUModel(FakeModel):
        def to(self, device):
            self.device_moves.append(str(device))
            if device.type == "cuda":
                raise RuntimeError("CUDA out of memory")
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = BusyGPUModel()
    predictor = SentimentPredictor(
        Settings(_env_file=None, device="auto"),
        load_metadata(metadata_path),
        tokenizer=fake_tokenizer,
        model=model,
    )
    assert model.device_moves == ["cuda", "cpu"]
    assert predictor.device == "cpu"
    assert predictor.predict("A good movie.").label in {"positive", "negative"}


def test_initialization_accepts_native_backend_when_token_contract_matches(
    metadata_path, fake_tokenizer, fake_model
) -> None:
    fake_tokenizer.is_fast = True
    predictor = make_predictor(metadata_path, fake_tokenizer, fake_model)

    assert predictor.tokenizer.is_fast is True


def test_probability_threshold_and_confidence_are_from_logits(
    metadata_path, fake_tokenizer
) -> None:
    model = FakeModel(
        [
            [0.0, logit_for_probability(0.4769)],
            [0.0, logit_for_probability(0.4771)],
        ]
    )
    predictor = make_predictor(metadata_path, fake_tokenizer, model)

    negative, positive = predictor.predict_batch(["first", "second"])

    assert negative.label == "negative"
    assert positive.label == "positive"
    assert positive.positive_probability == pytest.approx(0.4771, abs=1e-6)
    assert positive.negative_probability == pytest.approx(0.5229, abs=1e-6)
    assert positive.confidence == pytest.approx(0.5229, abs=1e-6)
    assert predictor._to_result(0.477).label == "positive"


def test_batching_chunks_by_inference_batch_size_and_disables_gradients(
    metadata_path, fake_tokenizer, fake_model
) -> None:
    predictor = make_predictor(metadata_path, fake_tokenizer, fake_model, batch_size=2)
    results = predictor.predict_batch(["one", "two", "three"])

    assert len(results) == 3
    assert fake_model.forward_calls == 2
    assert fake_model.grad_enabled == [False, False]
    assert fake_tokenizer.batch_tokenize_calls == 2
    assert all(call["pad_to_multiple_of"] == 8 for call in fake_tokenizer.pad_calls)


@pytest.mark.parametrize("texts", [[], [""], ["   "]])
def test_predictor_rejects_invalid_input_before_model_call(
    metadata_path, fake_tokenizer, fake_model, texts
) -> None:
    predictor = make_predictor(metadata_path, fake_tokenizer, fake_model)
    with pytest.raises(ValueError, match="non-empty"):
        predictor.predict_batch(texts)
    assert fake_model.forward_calls == 0


def test_predictor_sanitizes_invalid_logits(metadata_path, fake_tokenizer) -> None:
    predictor = make_predictor(
        metadata_path, fake_tokenizer, FakeModel([[float("nan"), 0.0]])
    )
    with pytest.raises(PredictionError, match="Inference failed"):
        predictor.predict("review")
