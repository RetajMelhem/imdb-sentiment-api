"""Run a manual smoke test against the configured production model."""

from __future__ import annotations

import time

from sentiment.config import Settings, load_configured_metadata
from sentiment.predictor import SentimentPredictor

EXAMPLES = [
    "This film was thoughtful, beautifully acted, and genuinely moving.",
    "A tedious mess with terrible dialogue; I regretted watching it.",
    "The performances were excellent, but the ending felt rushed.",
]


def main() -> None:
    settings = Settings()
    metadata = load_configured_metadata(settings)
    predictor = SentimentPredictor(settings=settings, metadata=metadata)
    started = time.perf_counter()
    results = predictor.predict_batch(EXAMPLES)
    elapsed_ms = (time.perf_counter() - started) * 1000
    for review, result in zip(EXAMPLES, results, strict=True):
        print(f"Review: {review}")
        print(
            f"  label={result.label} confidence={result.confidence:.6f} "
            f"p_positive={result.positive_probability:.6f} "
            f"p_negative={result.negative_probability:.6f}"
        )
    print(f"Batch latency: {elapsed_ms:.2f} ms ({predictor.device})")


if __name__ == "__main__":
    main()
