# v1.1.0 - Portable model and Docker release

The Docker image includes the real fine-tuned IMDB DeBERTa model, tokenizer,
audited metadata, and a checksum manifest. A fresh source checkout can build and
start with one command in PowerShell or CMD:

```console
docker compose up --build -d --wait
```

Open <http://localhost:8080/docs> to enter reviews and test predictions.
The first source build requires internet access. Inference uses the bundled
model and does not need a model download, token, host Python, or GPU.

## Windows release bundle

Download `imdb-sentiment-api-v1.1.0-windows-amd64.zip` from this release and
extract the complete ZIP. With Docker Desktop running Linux containers, run:

PowerShell:

```powershell
.\start.cmd
```

CMD:

```bat
start.cmd
```

The launcher checks the archive SHA-256, imports the image when needed, and
waits until the API is healthy. The bundle supports offline inference after
download and includes `release.json`, `model-manifest.json`, and checksums.
Use `PORT=8081` in a `.env` file if port 8080 is occupied.

## Model provenance and requirements

- Model: `retaj249/imdb-deberta-v3-sentiment`.
- Immutable model revision: `0cadd57a4ea34183f3cd4f14f2902a06cf73ac5e`.
- `model/manifest.json` records every artifact's filename, size, and SHA-256.
- Weights match the training metadata hash
  `e84d203742335db3bef1e7ff1e4b04aeb5cf4811b06e132a10fd96a1ab747ddd`.
- Linux/amd64 CPU image; the Windows launcher targets Windows x64 with Docker
  Desktop. Allow at least 4 GB of available Docker memory and several GB of disk.
- Model inference settings remain 384-token head-tail truncation, labels
  negative/positive, and positive threshold 0.477.

The Python base image and runtime dependency versions are pinned. Docker runs
as an unprivileged user; Compose uses a read-only filesystem and localhost port
8080. Large model files stay out of Git and travel inside the release image.

Validation details are recorded in `DOCKER_VALIDATION.md` in the source release.
The original 25,000-example benchmark is not rerun as part of release packaging.
