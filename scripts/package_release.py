"""Package a verified Linux/amd64 image for the Windows one-command launcher."""

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


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


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
        raise RuntimeError("The Windows bundle requires a Linux/amd64 image.")
    manifest_path = ROOT / "model" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = ROOT / "dist" / f"v{version}"
    bundle = destination / f"imdb-sentiment-api-v{version}-windows-amd64"
    bundle.mkdir(parents=True, exist_ok=True)
    archive = bundle / f"imdb-sentiment-api-{version}-linux-amd64.tar"
    if not args.reuse_archive:
        subprocess.run(
            ["docker", "image", "save", "--output", str(archive), image_tag], check=True
        )
    elif not archive.is_file():
        raise FileNotFoundError(
            "No image archive to reuse; run without --reuse-archive."
        )
    with tarfile.open(archive) as saved:
        docker_manifest = json.load(saved.extractfile("manifest.json"))[0]
        config_bytes = saved.extractfile(docker_manifest["Config"]).read()
        config = json.loads(config_bytes)
    if config["rootfs"]["diff_ids"] != image["RootFS"]["Layers"]:
        raise RuntimeError("Saved archive does not match the built image layers.")
    if config["config"] != image["Config"]:
        raise RuntimeError(
            "Saved archive does not match the built image configuration."
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
    shutil.copyfile(manifest_path, bundle / "model-manifest.json")
    release = {
        "schema_version": 1,
        "version": version,
        "source_commit": command("git", "rev-parse", "HEAD"),
        "image": image_tag,
        "image_id": image["Id"],
        "image_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "rootfs_layers": image["RootFS"]["Layers"],
        "platform": "linux/amd64",
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "model_revision": manifest["revision"],
    }
    (bundle / "release.json").write_text(
        json.dumps(release, indent=2) + "\n", encoding="utf-8"
    )
    checksums = "".join(
        f"{sha256(path)}  {path.name}\n"
        for path in sorted(bundle.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    (bundle / "SHA256SUMS.txt").write_text(checksums, encoding="utf-8")
    zip_path = destination / f"{bundle.name}.zip"
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as zipped:
        for path in sorted(bundle.iterdir()):
            zipped.write(path, arcname=f"{bundle.name}/{path.name}")
    if zip_path.stat().st_size >= 2 * 1024**3:
        raise RuntimeError("Release ZIP exceeds the GitHub asset size limit.")
    shutil.copyfile(manifest_path, destination / "model-manifest.json")
    shutil.copyfile(ROOT / "RELEASE_NOTES.md", destination / "RELEASE_NOTES.md")
    (destination / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in [zip_path, destination / "model-manifest.json"]
        ),
        encoding="utf-8",
    )
    print(f"Release packaged: {zip_path.name} ({zip_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
