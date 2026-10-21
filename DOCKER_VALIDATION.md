# Docker release verification - v1.1.0

Verified on 2026-10-21 on this Windows x64 host, using Docker Desktop Linux
containers. These checks establish the behavior below on this host; a second
physical PC and ARM execution have not been tested. NVIDIA GPU execution was
tested on this host's RTX 3060 Laptop GPU (6 GB VRAM).

## Runtime and artifact identity

- Docker Engine 29.8.0; Compose 5.5.1.
- Container Python 3.12.15; Torch 2.13.0+cu130 (CUDA 13.0); Transformers 5.20.2;
  SentencePiece 0.2.2.
- Python base image pinned by SHA-256 digest in `Dockerfile`.
- Runtime dependencies pinned in `requirements-docker.txt` and
  `requirements-cuda.txt`.
- Model repository: `retaj249/imdb-deberta-v3-sentiment`.
- Model revision: `0cadd57a4ea34183f3cd4f14f2902a06cf73ac5e`.
- All nine model snapshot files verified against `model/manifest.json` both
  locally and inside the running image. Weights, config, and SentencePiece
  hashes also match the original training provenance.
- Fine-tuned weights: 737,719,272 bytes, SHA-256
  `e84d203742335db3bef1e7ff1e4b04aeb5cf4811b06e132a10fd96a1ab747ddd`.

## Code and model checks

- Ruff lint and format checks passed; `git diff --check` passed.
- All 45 tests passed in the CUDA-capable Linux image, including the two
  real-model integration tests and automatic GPU/CPU selection/fallback tests;
  serving-code branch coverage was 87.93% (required: 80%).
- Real-model parity covered tokenizer IDs/masks, head-tail overflow handling,
  logits, probabilities, thresholding, and labels against the independently
  implemented notebook reference path.
- Integration checks proved one predictor construction and true batch
  chunking for nine reviews with inference batch size eight.
- `pip check` found no broken requirements.
- `pip-audit` reported zero known vulnerabilities for all 49 pinned runtime
  packages and all 19 pinned CUDA dependencies. Separate queries for pip 26.2.2
  and Torch's public base version 2.13.0 also reported none. This is a
  point-in-time Python dependency audit, not an OS image audit.
- Candidate source files were scanned for common token/private-key formats;
  none were found. Model binaries and local caches remain ignored by Git.

## API and container checks

The default image was built with Docker Compose. The source `.\start.cmd`
launcher detected the installed image, successfully probed CUDA, enabled the GPU
Compose override, and returned a healthy service at port 8080. Use `-Rebuild`
after source edits; normal source starts reuse the installed image.

- `/` reported application version 1.1.0.
- `/health` returned `model_loaded: true`, `device: cuda` on the RTX 3060.
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

A separate container ran the CUDA-capable image with `--gpus all --network none
--read-only` and no model/cache mounts. It loaded actual model parameters on
`cuda:0`, predicted five representative texts (including HTML and token-budget
overflow), moved the same model parameters to CPU, and predicted again.
All five labels matched; the maximum GPU/CPU positive-probability difference
was `2.384185791015625e-07`. The observed positive probability for the short
integration positive text was `0.9954326152801514` through the GPU API.
This establishes real GPU and CPU inference without Hub/network access.

## Windows portable archive and clean import

The final CUDA/CPU image was exported as a Docker-compatible OCI archive,
streamed directly from containerd to avoid an additional expanded export copy
on this host's nearly full disk. It contains four archive parts totaling
3,564,425,216 bytes. The packaging verifier compared the saved configuration
and all layer identities with the built image. The complete archive SHA-256 is
`b1bdc53095482bbe02a470ea3827d468c8fdc18c50c26a9dba70860aac15e9f9`.
`release.json` records each part's size/hash, the configuration digest, layer
identities, and source/model provenance; bundle and ZIP checksums are separate.

The Windows release launcher was exercised through `start.cmd`, including
Windows PowerShell 5.1, from this checkout's path containing spaces and Arabic
characters. The archive checksum uses .NET directly, so it does not depend on
PowerShell module search paths.

A disposable Docker-in-Docker 29.8.0 engine initially had no images or volumes.
The launcher verified all four local parts and the complete archive, streamed
them into Docker, and confirmed the imported layer identities. This engine
had no GPU access: the automatic probe failed and the launcher selected CPU
and system RAM. Compose used `--no-build --pull never`, started the API, and
waited for health. `/health` reported `model_loaded: true`, `device: cpu`.

The imported service ran as UID 999 with a read-only filesystem, no mounts,
no model cache volumes, and no GPU device requests. All nine embedded model
files passed size/SHA-256 verification. Real positive and negative reviews
produced the expected labels; a nine-review batch returned nine predictions
with single/batch parity within 0.000001, and empty input returned HTTP 422.
Rerunning the same Windows launcher reused the installed image and returned
health successfully without another import or download. This final multipart
test supplements the earlier CPU package clean-import checks.

The offline/local-part installation path was tested. Public GitHub downloads
cannot be tested before publication; the uploaded draft assets' sizes and
GitHub SHA-256 digests match all four local archive parts. Docker Desktop's
Windows engine connection stalled after the earlier disk-full export; direct
WSL engine access was used for export. The clean import used the Windows
Docker CLI against the disposable engine. Restarting Docker Desktop with a
fresh WSL kernel restored its normal Windows connection.

The same four-part release was then imported into Docker Desktop through its
normal Windows connection. The release `start.cmd` detected the RTX 3060,
enabled the GPU override, and returned healthy with `device: cuda`. The root
source `start.cmd` also reused that imported image successfully. A real
positive API prediction passed after restoration. The local GPU service was
left running at port 8080, and the disposable engine and its volume were removed.

## Limits

The original 25,000-example benchmark and optional recovered validation
prediction archive were not rerun. Existing benchmark values remain historical
training results. Deprecation warnings from upstream test-client/JIT APIs did
not fail the tests. The `/docs` HTML endpoint works, but Swagger UI assets use
a CDN; the first uncached browser visit requires internet. Direct HTTP
inference works offline as verified above.
