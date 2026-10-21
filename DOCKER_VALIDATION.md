# Docker release verification - v1.1.0

Verified on 2026-10-21 on this Windows x64 host, using Docker Desktop Linux
containers. These checks establish the behavior below on this host; a second
physical PC, ARM, and GPU execution have not been tested.

## Runtime and artifact identity

- Docker Engine 29.8.0; Compose 5.5.1.
- Container Python 3.12.15; Torch 2.13.0+cpu; Transformers 5.20.2;
  SentencePiece 0.2.2.
- Python base image pinned by SHA-256 digest in `Dockerfile`.
- Runtime dependencies pinned in `requirements-docker.txt`.
- Model repository: `retaj249/imdb-deberta-v3-sentiment`.
- Model revision: `0cadd57a4ea34183f3cd4f14f2902a06cf73ac5e`.
- All nine model snapshot files verified against `model/manifest.json` both
  locally and inside the running image. Weights, config, and SentencePiece
  hashes also match the original training provenance.
- Fine-tuned weights: 737,719,272 bytes, SHA-256
  `e84d203742335db3bef1e7ff1e4b04aeb5cf4811b06e132a10fd96a1ab747ddd`.

## Code and model checks

- Ruff lint and format checks passed; `git diff --check` passed.
- All 42 tests passed in the final Linux image, including the two real-model
  integration tests; serving-code branch coverage was 88.06% (required: 80%).
- Real-model parity covered tokenizer IDs/masks, head-tail overflow handling,
  logits, probabilities, thresholding, and labels against the independently
  implemented notebook reference path.
- Integration checks proved one predictor construction and true batch
  chunking for nine reviews with inference batch size eight.
- `pip check` found no broken requirements.
- `pip-audit` reported zero known vulnerabilities for all 49 pinned runtime
  packages. Separate queries for pip 26.2.2 and Torch's public base version
  2.13.0 also reported none. The installed Torch wheel is the official CPU
  variant. This is a point-in-time Python dependency audit, not an OS image audit.
- Candidate source files were scanned for common token/private-key formats;
  none were found. Model binaries and local caches remain ignored by Git.

## API and container checks

The documented `docker compose up --build -d --wait` path built the image,
started the actual model, and returned a healthy service at port 8080.

- `/` reported application version 1.1.0.
- `/health` returned `model_loaded: true`, `device: cpu`.
- `/model-info` preserved threshold 0.477 and token budget 384.
- Positive and negative reviews produced their expected labels.
- A nine-review batch returned nine predictions, including the expected first
  positive/negative pair; the matching single/batch probability difference was
  less than 0.000001.
- Empty review input returned HTTP 422.
- The README's PowerShell `Invoke-RestMethod` and CMD `curl.exe` examples
  both produced a real positive prediction.
- Running UID was 999. Compose used a read-only root filesystem, no mounts,
  localhost port binding, dropped capabilities, and no-new-privileges.

A separate container started with `--network none --read-only`, no mounts, and
the same image. Startup, health, single inference, nine-review batch inference,
and invalid-input validation all passed. For the integration positive text,
the observed positive probability was `0.9954326152801514`, matching the cold
engine result below. This proves inference does not require Hub access or a
host model/cache directory after the image is built.

## Windows portable archive and clean import

The image was exported with `docker image save`. The exporter compared the
saved archive's layer identities and image configuration with the built image.
Archive SHA-256 and source/model provenance are recorded in `release.json`;
bundle and ZIP checksums are provided separately.

The extracted Windows launcher was exercised through `start.cmd`, including
Windows PowerShell 5.1, from this checkout's path containing spaces and Arabic
characters. The archive checksum uses .NET directly, so it does not depend on
PowerShell module search paths.

A disposable Docker-in-Docker 29.8.0 engine initially had no images or volumes.
The launcher verified and imported the complete image archive, used
`--no-build --pull never`, started Compose, and waited for health. The cold
service had no mounts or model cache volumes. Its real single prediction and
nine-review batch passed, and rerunning the launcher reused the installed
image successfully.

## Limits

The original 25,000-example benchmark and optional recovered validation
prediction archive were not rerun. Existing benchmark values remain historical
training results. Deprecation warnings from upstream test-client/JIT APIs did
not fail the tests. The `/docs` HTML endpoint works, but Swagger UI assets use
a CDN; the first uncached browser visit requires internet. Direct HTTP
inference works offline as verified above.
