"""Export the complete CUDA/CPU image in verified GitHub-sized archive parts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PART_SIZE = 1024**3


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


class PartStream:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = iter(paths)
        self.stream = None

    def read(self, size: int) -> bytes:
        chunks = []
        while size:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.stream = path.open("rb")
            chunk = self.stream.read(size)
            if not chunk:
                self.stream.close()
                self.stream = None
                continue
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()


def export_parts(image_tag: str, destination: Path, version: str) -> list[Path]:
    process = subprocess.Popen(
        ["docker", "image", "save", image_tag], stdout=subprocess.PIPE
    )
    paths = []
    try:
        finished = False
        while not finished:
            path = destination / (
                f"imdb-sentiment-api-{version}-linux-amd64.tar.part{len(paths):03}"
            )
            size = 0
            with path.open("wb") as target:
                while size < PART_SIZE:
                    chunk = process.stdout.read(min(1024 * 1024, PART_SIZE - size))
                    if not chunk:
                        finished = True
                        break
                    target.write(chunk)
                    size += len(chunk)
            if size:
                paths.append(path)
                print(f"Exported {path.name} ({size:,} bytes)", flush=True)
            else:
                path.unlink()
        if process.wait() != 0:
            raise RuntimeError("Docker image export failed.")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    return paths


def inspect_archive(paths: list[Path], image: dict) -> str:
    candidates = {}
    docker_manifest = None
    stream = PartStream(paths)
    try:
        with tarfile.open(fileobj=stream, mode="r|") as saved:
            for member in saved:
                if not member.isfile() or member.size > 256 * 1024:
                    continue
                payload = saved.extractfile(member).read()
                try:
                    parsed = json.loads(payload)
                except (ValueError, UnicodeDecodeError):
                    continue
                if member.name == "manifest.json":
                    docker_manifest = parsed[0]
                elif isinstance(parsed, dict) and "rootfs" in parsed:
                    candidates[member.name] = (parsed, payload)
    finally:
        stream.close()
    if docker_manifest is None:
        raise RuntimeError("Saved image has no Docker archive manifest.")
    config, payload = candidates[docker_manifest["Config"]]
    if config["rootfs"]["diff_ids"] != image["RootFS"]["Layers"]:
        raise RuntimeError("Saved archive layers differ from the image.")
    if config["config"] != image["Config"]:
        raise RuntimeError("Saved archive configuration differs from the image.")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-archive", action="store_true")
    args = parser.parse_args()
    if command("git", "status", "--porcelain"):
        raise RuntimeError("Commit the reviewed source before packaging a release.")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Release version must be major.minor.patch.")
    image_tag = f"imdb-sentiment-api:{version}"
    image = json.loads(command("docker", "image", "inspect", image_tag))[0]
    if (image["Os"], image["Architecture"]) != ("linux", "amd64"):
        raise RuntimeError("The Windows release requires a Linux/amd64 image.")
    manifest_path = ROOT / "model" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = ROOT / "dist" / f"v{version}"
    bundle = destination / f"imdb-sentiment-api-v{version}-windows-amd64"
    bundle.mkdir(parents=True, exist_ok=True)
    if args.reuse_archive:
        paths = sorted(destination.glob(f"imdb-sentiment-api-{version}-*.part???"))
        if not paths:
            raise FileNotFoundError("No archive parts to reuse.")
    else:
        paths = export_parts(image_tag, destination, version)
    config_digest = inspect_archive(paths, image)
    archive_hash = hashlib.sha256()
    parts = []
    for path in paths:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                archive_hash.update(chunk)
        parts.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha256(path),
                "url": "https://github.com/RetajMelhem/imdb-sentiment-api/"
                f"releases/download/v{version}/{path.name}",
            }
        )
    for source in (ROOT / "packaging" / "windows").iterdir():
        shutil.copyfile(source, bundle / source.name)
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    build_block = "    build:\n      context: .\n"
    if compose.count(build_block) != 1:
        raise RuntimeError("Review Compose build settings before packaging.")
    (bundle / "compose.yaml").write_text(
        compose.replace(build_block, ""), encoding="utf-8"
    )
    shutil.copyfile(ROOT / "docker-compose.gpu.yml", bundle / "compose.gpu.yaml")
    shutil.copyfile(manifest_path, bundle / "model-manifest.json")
    release = {
        "schema_version": 2,
        "version": version,
        "source_commit": command("git", "rev-parse", "HEAD"),
        "image": image_tag,
        "image_id": image["Id"],
        "image_config_sha256": config_digest,
        "rootfs_layers": image["RootFS"]["Layers"],
        "platform": "linux/amd64",
        "device_priority": ["cuda", "cpu"],
        "archive_sha256": archive_hash.hexdigest(),
        "parts": parts,
        "model_revision": manifest["revision"],
    }
    (bundle / "release.json").write_text(
        json.dumps(release, indent=2) + "\n", encoding="utf-8"
    )
    (bundle / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in sorted(bundle.iterdir())
            if path.is_file() and path.name != "SHA256SUMS.txt"
        ),
        encoding="utf-8",
    )
    zip_path = destination / f"{bundle.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(bundle.iterdir()):
            zipped.write(path, arcname=f"{bundle.name}/{path.name}")
    shutil.copyfile(manifest_path, destination / "model-manifest.json")
    shutil.copyfile(ROOT / "RELEASE_NOTES.md", destination / "RELEASE_NOTES.md")
    (destination / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in [zip_path, destination / "model-manifest.json", *paths]
        ),
        encoding="utf-8",
    )
    print(f"Packaged starter ZIP and {len(parts)} verified image archive parts.")


if __name__ == "__main__":
    main()
