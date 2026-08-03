"""Domain exceptions exposed by the inference layer."""


class SentimentError(Exception):
    """Base exception for expected sentiment-service failures."""


class ModelConfigurationError(SentimentError):
    """Raised when no safe fine-tuned model source is configured."""


class ModelLoadError(SentimentError):
    """Raised when model artifacts cannot be initialized."""


class PredictionError(SentimentError):
    """Raised when an initialized model cannot produce a valid prediction."""


class InferenceBusyError(PredictionError):
    """Raised when bounded inference capacity is temporarily exhausted."""
