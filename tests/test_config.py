from pathlib import Path

import pytest
from pydantic import ValidationError

from sentiment.config import Settings, load_configured_metadata, load_metadata
from sentiment.exceptions import ModelConfigurationError


def test_metadata_contains_audited_notebook_contract(metadata_path: Path) -> None:
    metadata = load_metadata(metadata_path)

    assert metadata.base_model == "microsoft/deberta-v3-base"
    assert metadata.max_length == 384
    assert metadata.decision_threshold == pytest.approx(0.477)
    assert metadata.id2label == {0: "negative", 1: "positive"}
    assert metadata.positive_class_id == 1
    assert metadata.inference_batch_size == 8
    assert metadata.test_metrics["accuracy"] == pytest.approx(0.96088)
    assert metadata.final_seed_validation_metrics["accuracy"] == pytest.approx(
        0.954453849254333
    )
    assert metadata.final_seed_validation_metrics["f1"] == pytest.approx(
        0.9549621363092866
    )
    assert metadata.final_seed_validation_metrics["log_loss"] == pytest.approx(
        0.16217701312185426
    )
    assert metadata.final_seed_validation_metrics["roc_auc"] == pytest.approx(
        0.990061801002073
    )
    assert metadata.artifact_provenance.model_sha256 == (
        "e84d203742335db3bef1e7ff1e4b04aeb5cf4811b06e132a10fd96a1ab747ddd"
    )
    assert metadata.artifact_provenance.validation_predictions_sha256 == (
        "c0d78a423c518b84cdb2e0e2a54350cab8e5aa2b85374150d1044a150f4d68ef"
    )


def test_settings_local_path_takes_precedence(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    settings = Settings(
        _env_file=None,
        model_path=model_dir,
        model_id="owner/fine-tuned-model",
        device="cpu",
    )

    assert settings.model_source() == str(model_dir.resolve())


def test_settings_accepts_explicit_fine_tuned_hub_id() -> None:
    settings = Settings(
        _env_file=None,
        model_path=None,
        model_id="owner/imdb-fine-tuned",
        hf_revision="a" * 40,
    )
    assert settings.model_source() == "owner/imdb-fine-tuned"


def test_settings_prefers_model_colocated_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    colocated = model_dir / "model_metadata.json"
    colocated.write_text("{}", encoding="utf-8")

    settings = Settings(_env_file=None, model_path=model_dir)

    assert settings.resolved_metadata_path() == colocated.resolve()


def test_hub_metadata_uses_same_pinned_revision(
    metadata_path: Path, monkeypatch
) -> None:
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(metadata_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    settings = Settings(
        _env_file=None,
        model_path=None,
        model_id="owner/imdb-fine-tuned",
        hf_revision="b" * 40,
    )

    metadata = load_configured_metadata(settings)

    assert metadata.base_model == "microsoft/deberta-v3-base"
    assert captured["repo_id"] == settings.model_id
    assert captured["revision"] == settings.hf_revision
    assert captured["filename"] == "model_metadata.json"


def test_settings_never_falls_back_to_untrained_base_model() -> None:
    settings = Settings(_env_file=None, model_path=None, model_id=None)
    with pytest.raises(ModelConfigurationError, match="fine-tuned"):
        settings.model_source()


def test_settings_rejects_invalid_limits() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, inference_batch_size=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_batch_size=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, device="tpu")
