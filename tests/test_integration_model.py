"""Optional acceptance test for the recovered production checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from scripts.verify_parity import verify
from sentiment.config import Settings
from sentiment.predictor import SentimentPredictor


@pytest.mark.integration
def test_exported_checkpoint_matches_notebook() -> None:
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        pytest.skip("Set MODEL_PATH to run the real-model parity gate.")
    validation_path = os.getenv("VALIDATION_PREDICTIONS_PATH")
    report = verify(
        Path(model_path),
        atol=1e-6,
        validation_predictions_path=(
            Path(validation_path) if validation_path else None
        ),
    )

    assert report.example_count >= 5
    assert report.max_logit_absolute_difference <= 1e-6
    assert report.max_positive_probability_absolute_difference <= 1e-6
    assert report.max_negative_probability_absolute_difference <= 1e-6
    assert report.max_notebook_probability_absolute_difference <= 1e-4
    if validation_path:
        assert report.validation_predictions_verified is True


@pytest.mark.integration
def test_real_model_api_loads_once_and_performs_chunked_batch_inference() -> None:
    model_path_value = os.getenv("MODEL_PATH")
    if not model_path_value:
        pytest.skip("Set MODEL_PATH to run the real-model API gate.")
    model_path = Path(model_path_value)
    settings = Settings(
        _env_file=None,
        model_path=model_path,
        model_id=None,
        device="cpu",
        inference_batch_size=8,
    )
    factory_calls = 0
    forward_calls = 0
    hook_handle = None

    def factory(config, model_metadata):
        nonlocal factory_calls, forward_calls, hook_handle
        factory_calls += 1
        predictor = SentimentPredictor(config, model_metadata)

        def count_forward(_module, _inputs, _output) -> None:
            nonlocal forward_calls
            forward_calls += 1

        hook_handle = predictor.model.register_forward_hook(count_forward)
        return predictor

    app = create_app(
        settings=settings,
        predictor_factory=factory,
    )
    texts = [
        "This film was thoughtful, beautifully acted, and genuinely moving.",
        "A tedious mess with terrible dialogue; I regretted watching it.",
        *[f"A neutral test review number {index}." for index in range(7)],
    ]
    try:
        with TestClient(app) as client:
            assert client.get("/").status_code == 200
            assert client.get("/health").json() == {
                "status": "ok",
                "model_loaded": True,
                "device": "cpu",
            }
            model_info = client.get("/model-info")
            docs = client.get("/docs")
            batch = client.post("/predict/batch", json={"texts": texts})
            single = client.post("/predict", json={"text": texts[0]})
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    assert factory_calls == 1
    assert model_info.status_code == 200
    assert model_info.json()["threshold"] == 0.477
    assert docs.status_code == 200
    assert batch.status_code == 200
    predictions = batch.json()["predictions"]
    assert len(predictions) == 9
    assert predictions[0]["label"] == "positive"
    assert predictions[1]["label"] == "negative"
    assert single.status_code == 200
    assert single.json()["label"] == "positive"
    assert forward_calls == 3
