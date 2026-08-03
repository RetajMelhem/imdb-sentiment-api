"""FastAPI entrypoint with one-time model initialization."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.middleware import BodySizeLimitMiddleware
from sentiment.config import ModelMetadata, Settings, load_configured_metadata
from sentiment.exceptions import InferenceBusyError, PredictionError
from sentiment.logging_config import configure_logging
from sentiment.predictor import SentimentPredictor
from sentiment.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResult,
    RootResponse,
)

LOGGER = logging.getLogger(__name__)
PredictorFactory = Callable[[Settings, ModelMetadata], SentimentPredictor]


def create_app(
    settings: Settings | None = None,
    metadata: ModelMetadata | None = None,
    predictor_factory: PredictorFactory | None = None,
) -> FastAPI:
    """Create the service, allowing lightweight predictor injection in tests."""
    active_settings = settings or Settings()
    active_metadata = metadata or load_configured_metadata(active_settings)
    factory = predictor_factory or (
        lambda config, model_metadata: SentimentPredictor(
            settings=config,
            metadata=model_metadata,
        )
    )
    configure_logging(active_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        LOGGER.info("Starting sentiment API")
        application.state.predictor = factory(active_settings, active_metadata)
        LOGGER.info("Fine-tuned model loaded successfully")
        try:
            yield
        finally:
            application.state.predictor = None
            LOGGER.info("Sentiment API shutdown complete")

    application = FastAPI(
        title="Production-Ready Transformer Sentiment Analysis",
        version="1.0.0",
        description="Fine-tuned DeBERTa sentiment inference for IMDB reviews.",
        lifespan=lifespan,
    )
    application.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=active_settings.max_request_bytes,
    )

    @application.exception_handler(PredictionError)
    async def prediction_error_handler(_: Request, __: PredictionError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": "The model could not process this request."},
        )

    @application.exception_handler(InferenceBusyError)
    async def inference_busy_handler(
        _: Request, __: InferenceBusyError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "Inference capacity is temporarily busy."},
            headers={"Retry-After": "1"},
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request, error: RequestValidationError
    ) -> JSONResponse:
        safe_errors = [
            {
                "loc": item["loc"],
                "msg": item["msg"],
                "type": item["type"],
            }
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": safe_errors})

    @application.get("/", response_model=RootResponse)
    def root() -> RootResponse:
        return RootResponse(
            name=application.title,
            version=application.version,
            docs="/docs",
            health="/health",
            model_info="/model-info",
        )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        predictor = request.app.state.predictor
        return HealthResponse(
            status="ok",
            model_loaded=predictor is not None,
            device=predictor.device,
        )

    @application.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        source_type = "local" if active_settings.model_path else "huggingface-hub"
        return ModelInfoResponse(
            task=active_metadata.task,
            model=active_metadata.base_model,
            model_source=source_type,
            dataset=active_metadata.dataset,
            max_length=active_settings.effective_max_length(active_metadata),
            threshold=active_settings.effective_threshold(active_metadata),
            labels=[
                active_metadata.id2label[index]
                for index in sorted(active_metadata.id2label)
            ],
            preprocessing=active_metadata.preprocessing,
            truncation_strategy=active_metadata.truncation_strategy,
            test_metrics=active_metadata.test_metrics,
            frozen_config_fingerprint=active_metadata.frozen_config_fingerprint,
        )

    @application.post("/predict", response_model=PredictionResult)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResult:
        if len(payload.text) > active_settings.max_text_chars:
            raise HTTPException(status_code=422, detail="Review text is too long.")
        return request.app.state.predictor.predict(payload.text)

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        payload: BatchPredictionRequest, request: Request
    ) -> BatchPredictionResponse:
        if len(payload.texts) > active_settings.max_batch_size:
            raise HTTPException(status_code=422, detail="Batch size exceeds the limit.")
        if any(len(text) > active_settings.max_text_chars for text in payload.texts):
            raise HTTPException(status_code=422, detail="Review text is too long.")
        predictions = request.app.state.predictor.predict_batch(payload.texts)
        return BatchPredictionResponse(predictions=predictions)

    return application


app = create_app()
