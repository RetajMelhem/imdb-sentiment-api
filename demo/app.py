from __future__ import annotations

import gradio as gr

from sentiment.config import Settings, load_configured_metadata
from sentiment.predictor import SentimentPredictor


def create_predictor() -> SentimentPredictor:
    """Load the same production predictor and metadata used by FastAPI."""
    settings = Settings()
    return SentimentPredictor(
        settings=settings,
        metadata=load_configured_metadata(settings),
    )


predictor = create_predictor()


def analyze_review(review: str) -> tuple[dict[str, float], str]:
    if not review or not review.strip():
        raise gr.Error("Enter a non-empty movie review.")

    result = predictor.predict(review)
    probabilities = {
        "Positive": result.positive_probability,
        "Negative": result.negative_probability,
    }
    summary = f"**{result.label.title()}** - confidence: **{result.confidence:.2%}**"
    return probabilities, summary


demo = gr.Interface(
    fn=analyze_review,
    inputs=gr.Textbox(
        label="Movie review",
        lines=8,
        placeholder="Write or paste an IMDB-style review...",
    ),
    outputs=[
        gr.Label(label="Class probabilities", num_top_classes=2),
        gr.Markdown(label="Decision"),
    ],
    title="IMDB Transformer Sentiment Analysis",
    description=(
        "Analyze a review with the exported fine-tuned DeBERTa model. "
        "The UI uses the production predictor, including its audited 0.477 threshold."
    ),
    examples=[
        ["A moving, beautifully acted film that stayed with me for days."],
        ["A tedious script and lifeless performances made this painful to finish."],
    ],
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
