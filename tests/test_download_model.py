"""Integrity and offline guarantees for portable model imports."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import download_model


@pytest.fixture
def snapshot(tmp_path):
    payload = b"a verified artifact"
    expected = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {
        "schema_version": 1,
        "repo_id": "owner/model",
        "revision": "a" * 40,
        "files": dict.fromkeys(download_model.REQUIRED_FILES, expected),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest, payload


def test_import_pins_all_downloads_and_verifies_offline(
    snapshot, tmp_path, monkeypatch
):
    path, manifest, payload = snapshot
    urls = []

    def open_artifact(request, timeout):
        urls.append(request.full_url)
        return io.BytesIO(payload)

    monkeypatch.setattr(download_model.urllib.request, "urlopen", open_artifact)
    output = tmp_path / "model"
    download_model.prepare_model(path, output, verify_only=False)
    assert len(urls) == len(manifest["files"])
    assert all(f"/resolve/{manifest['revision']}/" in url for url in urls)
    assert json.loads((output / "manifest.json").read_text()) == manifest

    def no_network(*args, **kwargs):
        pytest.fail("A verified snapshot must work without network access.")

    monkeypatch.setattr(download_model.urllib.request, "urlopen", no_network)
    download_model.prepare_model(path, output, verify_only=True)
    download_model.prepare_model(path, output, verify_only=False)
    artifact = output / "model.safetensors"
    artifact.write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="corrupt"):
        download_model.prepare_model(path, output, verify_only=True)


def test_corrupt_download_never_replaces_existing_artifact(tmp_path, monkeypatch):
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"original")
    monkeypatch.setattr(
        download_model.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad")
    )
    monkeypatch.setattr(download_model.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="mismatch"):
        download_model.download(
            "https://huggingface.co/owner/model/resolve/commit/model.safetensors",
            target,
            {"size": 3, "sha256": hashlib.sha256(b"yes").hexdigest()},
        )
    assert target.read_bytes() == b"original"
    assert not target.with_name(target.name + ".part").exists()


@pytest.mark.parametrize("failure", ["mutable_revision", "path_traversal", "missing"])
def test_rejects_unreproducible_or_incomplete_manifest(snapshot, failure):
    path, manifest, _ = snapshot
    if failure == "mutable_revision":
        manifest["revision"] = "main"
    elif failure == "path_traversal":
        manifest["files"]["../outside"] = manifest["files"]["config.json"]
    else:
        del manifest["files"]["model.safetensors"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        download_model.load_manifest(path)


def test_repository_manifest_matches_training_provenance():
    root = Path(__file__).resolve().parents[1]
    manifest = download_model.load_manifest(root / "model" / "manifest.json")
    metadata = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
    provenance = metadata["artifact_provenance"]
    for filename, key in [
        ("config.json", "config_sha256"),
        ("model.safetensors", "model_sha256"),
        ("spm.model", "tokenizer_spm_sha256"),
    ]:
        assert manifest["files"][filename]["sha256"] == provenance[key]
