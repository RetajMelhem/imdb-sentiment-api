from pathlib import Path

import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.export_model import export_model


def test_export_model_writes_standard_artifacts(
    tmp_path: Path,
    metadata_path: Path,
    fake_model,
    fake_tokenizer,
    monkeypatch,
) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")
    tokenizer_artifacts = {
        "spm.model": b"sentencepiece",
        "added_tokens.json": b"{}",
        "special_tokens_map.json": b"{}",
        "tokenizer_config.json": b'{"tokenizer_class":"DebertaV2Tokenizer"}',
        "tokenizer.json": b'{"version":"1.0"}',
    }
    for name, content in tokenizer_artifacts.items():
        (source / name).write_bytes(content)
    output = tmp_path / "model"
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: fake_tokenizer,
    )
    monkeypatch.setattr(
        AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: fake_model,
    )

    exported = export_model(source, output, metadata_path)

    assert exported == output.resolve()
    assert (output / "config.json").is_file()
    assert (output / "model.safetensors").is_file()
    assert (output / "tokenizer_config.json").is_file()
    assert (output / "model_metadata.json").read_bytes() == metadata_path.read_bytes()
    assert (output / "config.json").read_bytes() == (
        source / "config.json"
    ).read_bytes()
    for name, content in tokenizer_artifacts.items():
        assert (output / name).read_bytes() == content


def test_export_model_rejects_missing_weights(
    tmp_path: Path, metadata_path: Path
) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="weights"):
        export_model(source, tmp_path / "model", metadata_path)


def test_export_model_protects_nonempty_output(
    tmp_path: Path, metadata_path: Path
) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")
    output = tmp_path / "model"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        export_model(source, output, metadata_path)


def test_export_model_rejects_overlapping_source_and_output(
    tmp_path: Path, metadata_path: Path
) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="overlap"):
        export_model(source, source / "export", metadata_path)
    with pytest.raises(ValueError, match="overlap"):
        export_model(source, tmp_path, metadata_path, force=True)


def test_export_model_force_rejects_linked_output_without_touching_target(
    tmp_path: Path, metadata_path: Path
) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    output_link = tmp_path / "linked-output"
    try:
        output_link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this platform.")

    with pytest.raises(ValueError, match="symbolic links or junctions"):
        export_model(source, output_link, metadata_path, force=True)

    assert marker.read_text(encoding="utf-8") == "keep"
