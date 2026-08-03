from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.main import create_app
from app.middleware import BodySizeLimitMiddleware
from sentiment.config import Settings, load_metadata
from sentiment.schemas import PredictionResult


class FakePredictor:
    def __init__(self) -> None:
        self.device = "cpu"
        self.predict_calls = 0
        self.batch_calls = 0

    @staticmethod
    def _result(label: str = "positive") -> PredictionResult:
        positive = 0.9 if label == "positive" else 0.1
        return PredictionResult(
            label=label,
            label_id=1 if label == "positive" else 0,
            confidence=max(positive, 1 - positive),
            positive_probability=positive,
            negative_probability=1 - positive,
        )

    def predict(self, text: str) -> PredictionResult:
        self.predict_calls += 1
        return self._result()

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        self.batch_calls += 1
        return [
            self._result("negative" if "hated" in text else "positive")
            for text in texts
        ]


def test_api_lifespan_loads_once_and_serves_all_endpoints(metadata_path) -> None:
    predictor = FakePredictor()
    factory_calls = []
    settings = Settings(
        _env_file=None,
        model_path=None,
        model_id="owner/fine-tuned",
        max_batch_size=2,
    )
    app = create_app(
        settings=settings,
        metadata=load_metadata(metadata_path),
        predictor_factory=lambda *_: factory_calls.append(True) or predictor,
    )

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        health = client.get("/health").json()
        model_info = client.get("/model-info").json()
        single = client.post("/predict", json={"text": "Excellent movie"})
        batch = client.post(
            "/predict/batch",
            json={"texts": ["Excellent movie", "I hated this film"]},
        )

    assert factory_calls == [True]
    assert health == {"status": "ok", "model_loaded": True, "device": "cpu"}
    assert model_info["model"] == "microsoft/deberta-v3-base"
    assert model_info["threshold"] == 0.477
    assert single.status_code == 200
    assert batch.status_code == 200
    assert batch.json()["predictions"][1]["label"] == "negative"
    assert predictor.predict_calls == 1
    assert predictor.batch_calls == 1


def test_api_rejects_blank_text_and_oversized_batch(metadata_path) -> None:
    predictor = FakePredictor()
    settings = Settings(
        _env_file=None,
        model_path=None,
        model_id="owner/fine-tuned",
        max_batch_size=1,
    )
    app = create_app(
        settings=settings,
        metadata=load_metadata(metadata_path),
        predictor_factory=lambda *_: predictor,
    )
    with TestClient(app) as client:
        blank = client.post("/predict", json={"text": "   "})
        response = client.post("/predict/batch", json={"texts": ["one", "two"]})
    assert blank.status_code == 422
    assert "input" not in blank.text
    assert "   " not in blank.text
    assert response.status_code == 422
    assert predictor.batch_calls == 0


def test_api_rejects_raw_oversized_body_and_unknown_fields(metadata_path) -> None:
    predictor = FakePredictor()
    settings = Settings(
        _env_file=None,
        model_path=None,
        model_id="owner/fine-tuned",
        max_request_bytes=1024,
    )
    app = create_app(
        settings=settings,
        metadata=load_metadata(metadata_path),
        predictor_factory=lambda *_: predictor,
    )
    with TestClient(app) as client:
        oversized = client.post(
            "/predict",
            json={"text": "ok", "ignored": "x" * 2000},
        )
        unknown = client.post(
            "/predict",
            json={"text": "ok", "ignored": "small"},
        )

    assert oversized.status_code == 413
    assert len(oversized.content) < 100
    assert unknown.status_code == 422
    assert predictor.predict_calls == 0


def test_hub_startup_pins_metadata_tokenizer_and_model_to_same_revision(
    metadata_path, fake_tokenizer, fake_model, monkeypatch
) -> None:
    revision = "c" * 40
    calls = []

    def fake_download(**kwargs):
        calls.append(("metadata", kwargs["repo_id"], kwargs["revision"]))
        return str(metadata_path)

    def fake_tokenizer_load(source, **kwargs):
        calls.append(("tokenizer", source, kwargs["revision"]))
        return fake_tokenizer

    def fake_model_load(source, **kwargs):
        calls.append(("model", source, kwargs["revision"]))
        return fake_model

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_tokenizer_load)
    monkeypatch.setattr(
        AutoModelForSequenceClassification,
        "from_pretrained",
        fake_model_load,
    )
    settings = Settings(
        _env_file=None,
        model_path=None,
        model_id="owner/imdb-fine-tuned",
        hf_revision=revision,
        device="cpu",
    )

    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/health").status_code == 200

    assert calls == [
        ("metadata", settings.model_id, revision),
        ("tokenizer", settings.model_id, revision),
        ("model", settings.model_id, revision),
    ]


def test_body_limit_rejects_chunked_stream_without_content_length() -> None:
    async def consume(scope, receive, send) -> None:
        more_body = True
        while more_body:
            message = await receive()
            more_body = message.get("more_body", False)
        await PlainTextResponse("ok")(scope, receive, send)

    async def exercise() -> list[dict]:
        messages = iter(
            [
                {"type": "http.request", "body": b"a" * 600, "more_body": True},
                {"type": "http.request", "body": b"b" * 600},
            ]
        )
        sent = []

        async def receive() -> dict:
            return next(messages)

        async def send(message: dict) -> None:
            sent.append(message)

        middleware = BodySizeLimitMiddleware(consume, max_bytes=1024)
        await middleware(
            {"type": "http", "method": "POST", "headers": [], "path": "/"},
            receive,
            send,
        )
        return sent

    sent = asyncio.run(exercise())
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
