"""Download or verify the immutable model snapshot using only Python's stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "model" / "manifest.json"
REQUIRED_FILES = {
    "config.json",
    "model.safetensors",
    "model_metadata.json",
    "spm.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
}


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported model manifest schema.")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", manifest.get("repo_id", "")):
        raise ValueError("Manifest must name a Hugging Face model repository.")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Manifest revision must be an immutable commit SHA.")
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not REQUIRED_FILES.issubset(files):
        raise ValueError("Manifest is missing required model or tokenizer files.")
    for name, expected in files.items():
        if not re.fullmatch(r"[\w][\w.-]*", name) or name in {".", ".."}:
            raise ValueError(f"Unsafe artifact filename: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected.get("sha256", "")):
            raise ValueError(f"Missing SHA-256 for {name}")
        if type(expected.get("size")) is not int or expected["size"] <= 0:
            raise ValueError(f"Invalid artifact size for {name}")
    return manifest


def matches(path: Path, expected: dict) -> bool:
    if not path.is_file() or path.stat().st_size != expected["size"]:
        return False
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest == expected["sha256"]


def download(url: str, target: Path, expected: dict) -> None:
    partial = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "imdb-sentiment-api-model-import/1.0"}
            )
            with (
                urllib.request.urlopen(request, timeout=90) as response,
                partial.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            if not matches(partial, expected):
                raise ValueError(f"Size or SHA-256 mismatch for {target.name}")
            partial.replace(target)
            return
        except (OSError, urllib.error.URLError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            print(f"Retrying {target.name} (attempt {attempt + 2}/3)", flush=True)
            time.sleep(2**attempt)


def prepare_model(manifest_path: Path, output: Path, *, verify_only: bool) -> None:
    manifest = load_manifest(manifest_path)
    if not verify_only:
        output.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files"].items():
        target = output / name
        if matches(target, expected):
            print(f"Verified {name}", flush=True)
            continue
        if verify_only:
            raise ValueError(f"Missing or corrupt model artifact: {name}")
        print(f"Downloading {name} ({expected['size']:,} bytes)", flush=True)
        url = (
            f"https://huggingface.co/{manifest['repo_id']}/resolve/"
            f"{manifest['revision']}/{name}"
        )
        download(url, target, expected)
        print(f"Verified {name}", flush=True)
    destination = output / "manifest.json"
    if not verify_only and manifest_path.resolve() != destination.resolve():
        shutil.copyfile(manifest_path, destination)
    print(f"Model ready (revision {manifest['revision']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=Path("model"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    prepare_model(args.manifest, args.output, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
