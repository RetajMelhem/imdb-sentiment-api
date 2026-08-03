"""Export a notebook-trained checkpoint as standard Hugging Face artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from sentiment.config import DEFAULT_METADATA_PATH, load_metadata

PRESERVED_ARTIFACTS = (
    "config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "spm.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _validate_source(source: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {source}")
    if not (source / "config.json").is_file():
        raise FileNotFoundError("Checkpoint is missing config.json")
    if not any(source.glob("model*.safetensors")):
        raise FileNotFoundError("Checkpoint is missing safetensors model weights")


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", lambda: False)
    return path.is_symlink() or bool(is_junction())


def _reject_linked_output_path(output: Path) -> None:
    for candidate in (output, *output.parents):
        if _is_link_or_junction(candidate):
            raise ValueError(
                "Output paths must not contain symbolic links or junctions."
            )


def _prepare_output(output: Path, force: bool) -> None:
    _reject_linked_output_path(output)
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output}. Pass --force to replace it."
            )
        if output == Path(output.anchor) or output == Path.cwd().resolve():
            raise ValueError("Refusing to replace a filesystem or project root.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def export_model(
    source: str | Path,
    output: str | Path,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    *,
    force: bool = False,
) -> Path:
    """Validate, reload, and save a fine-tuned checkpoint in HF format."""
    source_path = Path(source).expanduser().resolve()
    output_path = Path(os.path.abspath(Path(output).expanduser()))
    _reject_linked_output_path(output_path)
    resolved_output_path = output_path.resolve(strict=False)
    if source_path == resolved_output_path:
        raise ValueError("Source and output directories must be different.")
    if resolved_output_path.is_relative_to(source_path) or source_path.is_relative_to(
        resolved_output_path
    ):
        raise ValueError("Source and output directories must not overlap.")
    _validate_source(source_path)
    if output_path.exists() and any(output_path.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is not empty: {output_path}. Pass --force to replace it."
        )
    metadata = load_metadata(metadata_path)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source_path,
        use_fast=metadata.tokenizer.use_fast,
        trust_remote_code=False,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        source_path,
        trust_remote_code=False,
        local_files_only=True,
        use_safetensors=True,
    )
    _prepare_output(output_path, force)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    for artifact_name in PRESERVED_ARTIFACTS:
        source_artifact = source_path / artifact_name
        if source_artifact.is_file():
            shutil.copy2(source_artifact, output_path / artifact_name)
    shutil.copy2(Path(metadata_path).resolve(), output_path / "model_metadata.json")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported = export_model(
        args.source,
        args.output,
        args.metadata,
        force=args.force,
    )
    print(f"Exported fine-tuned model to: {exported}")


if __name__ == "__main__":
    main()
