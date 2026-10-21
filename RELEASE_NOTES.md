# v1.1.0 - GPU-first portable model and Docker release

The image includes the actual fine-tuned IMDB DeBERTa model, tokenizer, audited
metadata, CUDA runtime, and checksum manifest. GPU is the priority; when no
usable NVIDIA GPU is available, the same model loads on CPU and system RAM.
`/health` reports the actual device.

## One command from PowerShell or CMD

From a source checkout with Docker Desktop running Linux containers:

```console
.\start.cmd
```

The source launcher builds the image and probes GPU access through Docker.
The first build requires internet. Windows GPU support requires the WSL 2
backend and a compatible NVIDIA driver. CPU fallback works without a GPU.

## Windows release starter

Download `imdb-sentiment-api-v1.1.0-windows-amd64.zip`, extract it, and run:

```console
.\start.cmd
```

The starter downloads the `.tar.partNNN` assets, verifies each part and the
complete archive SHA-256, streams the image into Docker, probes GPU support,
starts the API, and waits for health. Each image part is under GitHub's per-file
size limit. No host Python, Git, Hugging Face token, or separate model setup is
needed. Allow several GB of downloads, about 20 GB free disk, and at least 4 GB
available Docker memory.

For offline installation, download all `.tar.partNNN` assets into the extracted
starter folder before running the same command. Subsequent starts reuse the
installed image. Inference requires no internet. The interactive `/docs` page
uses CDN-hosted Swagger assets; its first uncached browser visit needs internet.

Open <http://localhost:8080/docs> or use `curl.exe http://localhost:8080/health`.
Set `PORT=8081` in `.env` if port 8080 is occupied. `DEVICE=cpu` forces CPU.

## Model provenance

- Model: `retaj249/imdb-deberta-v3-sentiment`.
- Immutable revision: `0cadd57a4ea34183f3cd4f14f2902a06cf73ac5e`.
- `model/manifest.json` records every artifact's filename, size, and SHA-256.
- Weights match training hash
  `e84d203742335db3bef1e7ff1e4b04aeb5cf4811b06e132a10fd96a1ab747ddd`.
- The inference contract remains 384-token head-tail truncation, labels
  negative/positive, and positive threshold 0.477.

The Python base and runtime versions are pinned. The service runs as an
unprivileged user with a read-only Compose filesystem and localhost port 8080.
Large model files stay out of Git and are included in the release image.

See `DOCKER_VALIDATION.md` in the source release for measured GPU/CPU checks
and remaining limits. The original 25,000-example benchmark is not rerun for
release packaging.
